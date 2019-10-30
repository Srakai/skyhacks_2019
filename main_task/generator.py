import imgaug
from imgaug import augmenters
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras import backend as K
from tqdm import tqdm
import tensorflow as tf
import glob
import math
import pandas as pd
import cv2
import os
from utils import dbg
import data_preparation
import numpy as np


class DataGenerator(Sequence):
    def __init__(
        self,
        data_path,
        path_to_labels_file,
        data_shapes,
        batch_size,
        max_size="",
        is_prepared=False,
        tf_record_file_path="",
        augment_data=False,
        validation_split=0.0,
    ):
        """
        :param data_path: Path to image data
        :param path_to_labels_file: Path to csv file with image labels
        :param batch_size: Batch size
        :param max_size: The maximum value of images used in the generator. Optional parameter.
        :param is_prepared: If value is False, generator creates one file containing all images as tf_record file.
        :param tf_record_file_path: Path to tf_record file.
        :param augment_data: True if you want to perform augmentation on images.
        """
        if tf_record_file_path:
            self.tf_record_file_name = tf_record_file_path
        else:
            self.tf_record_file_name = data_path + "tf_record_dataset.tfrecord"

        self.image_shape = data_shapes["input"]
        self.label_shape = data_shapes["output"]
        self.augment_data = augment_data
        self.validation_split = validation_split
        # TODO change that
        self.images = data_preparation.get_data(data_path, data_preparation.folders) # + data_preparation.get_validation()

        self.labels = data_preparation.labels_generation(data_path, path_to_labels_file)

        # self.labels = self._load_image_labels(path_to_labels_file)
        self.batch_size = batch_size
        self.dataset_size = (
            min(len(self.images), max_size) if max_size else len(self.images)
        )

        if not is_prepared:
            self.prepare_dataset()
        dbg(f"Loaded number of entries: {self.dataset_size}")

        if self.augment_data:
            dbg(
                "Data augmentation won't work\n"
                "We need to change the imgaug to another library which is based on"
                "TF operation graph or invoke tf.session here which will cause lot"
                "of problems",
                mode="crit",
            )
            self.image_preprocessor = ImagePreprocessor()

        # create dataset object
        tfdataset = tf.data.TFRecordDataset(filenames=[self.tf_record_file_name])
        tfdataset = tfdataset.shuffle(256) # this might be a performace problem

        # if we want validation data, take 'validation_split' size out of dataset
        if self.validation_split:
            sizee = math.ceil(self.validation_split * self.dataset_size)
            self.train_data = self.images[:sizee]
            self.validation_data = self.images[sizee:]

            self.tfdataset_validation = tfdataset.take(
                math.ceil(self.validation_split * self.dataset_size)
            )
            self.tfdataset_validation = self.tfdataset_validation.map(
                self._parse_function
            )
            self.tfdataset_validation = self.tfdataset_validation.batch(self.batch_size)
            self.tfdataset_validation = self.tfdataset_validation.prefetch(
                tf.contrib.data.AUTOTUNE
            )
            self.tfdataset_validation = self.tfdataset_validation.repeat()
        else:
            self.train_data = self.images
        self.tfdataset_train = tfdataset.skip(
            math.ceil(self.validation_split * self.dataset_size)
        )
        self.tfdataset_train = self.tfdataset_train.map(self._parse_function)
        self.tfdataset_train = self.tfdataset_train.batch(self.batch_size)
        self.tfdataset_train = self.tfdataset_train.prefetch(tf.contrib.data.AUTOTUNE)
        self.tfdataset_train = self.tfdataset_train.repeat()

    def prepare_dataset(self):
        dbg("Preparing dataset")
        with tf.python_io.TFRecordWriter(self.tf_record_file_name) as writer:
            for i, entry in enumerate(tqdm(self.images, total=self.dataset_size)):
                # load and preprocess
                image_data = cv2.imread(entry)
                image_data = cv2.resize(image_data, (128, 128), 3)
                image_data = preprocess_input(image_data)


                # cut path and extention to get id
                image_id = os.path.basename(entry)#.replace(".jpg", "")
                try:
                    label = self.labels[image_id]
                except KeyError:
                    print(f'{image_id} does not exisit')
                    continue
                data = {
                    "image_data": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_data.tostring()])
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label)
                    ),
                }
                # Cast to TensorFlow Features.
                feature = tf.train.Features(feature=data)
                # Cast as a TensorFlow Example.
                example = tf.train.Example(features=feature)
                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)
                # if we reach desired dataset size, break preparation
                if i == self.dataset_size:
                    break
        dbg(f"Dataset file ({self.tf_record_file_name}) ready")

    def fetch_train_iterators(self):
        train_iterator = self.tfdataset_train.make_one_shot_iterator()
        fetch_val = train_iterator.get_next()
        with K.get_session().as_default() as sess:
            while True:
                *inputs, outputs = sess.run(fetch_val)
                yield inputs, outputs

    def fetch_validation_iterators(self):
        """
        Work is still in progress
        """
        if not self.validation_split:
            dbg("Validation split was not set/ set to 0", mode="crit")
        validation_iterator = self.tfdataset_validation.make_one_shot_iterator()
        fetch_val = validation_iterator.get_next()
        with K.get_session().as_default() as sess:
            while True:
                *inputs, outputs = sess.run(fetch_val)
                yield inputs, outputs

    def get_steps_per_epoch(self):
        return math.ceil(
            math.ceil(self.dataset_size * (1 - self.validation_split)) / self.batch_size
        )

    def get_validation_steps_per_epoch(self):
        return math.ceil(
            math.ceil(self.dataset_size * self.validation_split) / self.batch_size
        )

    def _parse_function(self, serialized):
        features = {
            "image_data": tf.FixedLenFeature([], tf.string),
            "label": tf.VarLenFeature(tf.int64),
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(
            serialized=serialized, features=features
        )

        # Get the image as raw bytes. 32 bits holds one byte - need to change that
        image_data = tf.decode_raw(parsed_example["image_data"], tf.int32)
        # We can change it to uint8 propably, but i doubt it will change performance on 64 bit cpu
        label = parsed_example["label"]#, tf.int64)

        if self.augment_data:
            image_data = self.image_preprocessor.augment_img(image_data=image_data)

        image_data = tf.reshape(image_data, self.image_shape)
        #label = tf.reshape(label, self.label_shape)

        image_data = tf.cast(image_data, tf.float32)
        #label = tf.cast(label, tf.float32)
        #tf.print(label)
        return image_data, label

    def _load_image_labels(self, path_to_labels_file):
        label_data = pd.read_csv(path_to_labels_file)
        labels = {
            ide: int(label) for ide, label in zip(label_data.id, label_data.label)
        }
        return labels

    def create_batch(self, seq, size):
        return (seq[pos: pos+size] for pos in range(0, len(seq), size))

    def fak_you_generator(self, validation=False):
        if validation:
            data = self.validation_data
        else:
            data = self.train_data
        while True:
            np.random.shuffle(data) # Shuffle dataset to generate different bunch of data in each iteration
            for batch in self.create_batch(data, self.batch_size):
                try:
                    Y = []
                    for i in batch:
                        i = os.path.basename(i)
                        Y.append(self.labels[i])
                except:
                    print(f'This image: {i} is not working!!!')
                    continue
                image_data = [cv2.imread(entry) for entry in batch]
                image_data = list(map(lambda x: cv2.resize(x, (128, 128), 3), image_data))
                image_data = list(map(preprocess_input, image_data))
                yield np.array(image_data), np.array(Y)


class ImagePreprocessor:
    def __init__(self):
        chance_var = lambda aug: augmenters.Sometimes(0.5, aug)

        self.augmenter_sequence = augmenters.Sequential(
            [
                augmenters.Fliplr(0.5),  # apply the following augmenters to most image
                augmenters.Flipud(0.2),  # horizontally flip 50% of all images
                # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                chance_var(
                    augmenters.CropAndPad(
                        percent=(-0.05, 0.1), pad_mode=imgaug.ALL, pad_cval=(0, 255)
                    )
                ),
                chance_var(
                    augmenters.Affine(
                        scale={
                            "x": (0.8, 1.2),
                            "y": (0.8, 1.2),
                        },  # scale images to 80-120% of their size,
                        # individually per axis
                        translate_percent={
                            "x": (-0.2, 0.2),
                            "y": (-0.2, 0.2),
                        },  # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or
                        # bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant,
                        # use a cval between 0 and 255
                        mode=imgaug.ALL  # use any of scikit-image's warping modes
                        # (see 2nd image from the top for examples)
                    )
                ),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                augmenters.SomeOf(
                    (0, 5),
                    [
                        # convert images into their superpixel representation
                        chance_var(
                            augmenters.Superpixels(
                                p_replace=(0, 1.0), n_segments=(20, 200)
                            )
                        ),
                        augmenters.OneOf(
                            [
                                # blur images with a sigma between 0 and 3.0
                                augmenters.GaussianBlur((0, 3.0)),
                                # blur image using local means with kernel sizes between 2 and 7
                                augmenters.AverageBlur(k=(2, 7)),
                                # blur image using local medians with kernel sizes between 2 and 7
                                augmenters.MedianBlur(k=(3, 11)),
                            ]
                        ),
                        augmenters.Sharpen(
                            alpha=(0, 1.0), lightness=(0.75, 1.5)
                        ),  # sharpen images
                        augmenters.Emboss(
                            alpha=(0, 1.0), strength=(0, 2.0)
                        ),  # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        augmenters.SimplexNoiseAlpha(
                            augmenters.OneOf(
                                [
                                    augmenters.EdgeDetect(alpha=(0.5, 1.0)),
                                    augmenters.DirectedEdgeDetect(
                                        alpha=(0.5, 1.0), direction=(0.0, 1.0)
                                    ),
                                ]
                            )
                        ),
                        # add gaussian noise to images
                        augmenters.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        augmenters.OneOf(
                            [
                                # randomly remove up to 10% of the pixels
                                augmenters.Dropout((0.01, 0.1), per_channel=0.5),
                                augmenters.CoarseDropout(
                                    (0.03, 0.15),
                                    size_percent=(0.02, 0.05),
                                    per_channel=0.2,
                                ),
                            ]
                        ),
                        # invert color channels
                        augmenters.Invert(0.05, per_channel=True),
                        # change brightness of images (by -10 to 10 of original value)
                        augmenters.Add((-10, 10), per_channel=0.5),
                        # change hue and saturation
                        augmenters.AddToHueAndSaturation((-20, 20)),
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        augmenters.OneOf(
                            [
                                augmenters.Multiply((0.5, 1.5), per_channel=0.5),
                                augmenters.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=augmenters.Multiply(
                                        (0.5, 1.5), per_channel=True
                                    ),
                                    second=augmenters.ContrastNormalization((0.5, 2.0)),
                                ),
                            ]
                        ),
                        # improve or worsen the contrast
                        augmenters.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                        augmenters.Grayscale(alpha=(0.0, 1.0)),
                        # move pixels locally around (with random strengths)
                        chance_var(
                            augmenters.ElasticTransformation(
                                alpha=(0.5, 3.5), sigma=0.25
                            )
                        ),
                        # sometimes move parts of the image around
                        chance_var(augmenters.PiecewiseAffine(scale=(0.01, 0.05))),
                        chance_var(augmenters.PerspectiveTransform(scale=(0.01, 0.1))),
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

    def augment_img(self, image_data):
        return self.augmenter_sequence.augment_image(image_data)
