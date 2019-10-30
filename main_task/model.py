"""
model python file contain three classes at this moment.
ModelPreparation():
    - contain methods necessary to preapere data and model to learning process

ModelCreation():
    - contain methods necessary to create model. In particular create_model() function.
CyclicR(Callback):
"""
import numpy as np
import generator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import cv2
import os
from utils import dbg
from glob import glob
import efficientnet.tfkeras as efn


class ModelCreation:
    def __init__(
        self, input_shape, output_shape, learning_rate, model_data_dir="./models/", saved_weights=False, img_shape=(128, 128, 3)
    ):
        """
        :param model_data_dir: Path where to store saved models etc
        """

        self.img_shape = img_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model_data_dir = model_data_dir

        dbg("Creating model graph")
        self.model = self.create_model()
        self.model.summary()

        if saved_weights:
            dbg(f"Loading saved weights: {saved_weights}")
            self.model.load_weights(saved_weights)

    def create_model(self):
        """
        input_tensor = Input(shape=self.input_shape)
        base_model = NASNetMobile(include_top=False, input_tensor=input_tensor)
        x = base_model(input_tensor)
        out_1 = GlobalMaxPooling2D()(x)
        out_2 = GlobalAveragePooling2D()(x)
        out_3 = Flatten()(x)
        out = Concatenate(axis=-1)([out_1, out_2, out_3])
        out = Dropout(0.5)(out)
        out = Dense(57, activation="sigmoid")(out)
        model = Model(input_tensor, out)
        model.compile(optimizer=Adam(self.learning_rate), loss=binary_crossentropy, metrics=['acc'])

        """
        # Efficientnet
        input_tensor = Input(shape=self.input_shape)
        base_model = efn.EfficientNetB3(weights='imagenet', include_top=True, input_tensor=input_tensor)
        x = base_model(input_tensor)
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        out = Dense(62, activation="sigmoid")(x)
        model = Model(input_tensor, out)
        model.compile(optimizer=SGD(self.learning_rate), loss='binary_crossentropy', metrics=['acc'])

        return model

    def save_model(self, filename="model"):
        model_file_path = self.model_data_dir + filename
        i = 1
        while os.path.isfile(model_file_path):
            model_file_path = self.model_data_dir + filename + "-" + str(i)
            i += 1
        weights_path = model_file_path + ".hdf5"
        self.model.save(model_file_path)
        self.model.save_weights(weights_path)
        dbg(f"model saved: {model_file_path}\tweights saved: {weights_path}")

    def evaluate(self, images):
        dbg("Evaluating the model on {} entries".format(len(images)))

        results = {}
        for img in images:
            try:
                image = cv2.imread(img)
                image = cv2.resize(image, (128, 128), 3)
            except:
                print(f"this image: {img} is no valid, ommiting..")
                continue

            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            results[img] = self.model.predict(image)
        return results
