from generator import DataGenerator
from utils import dbg, get_image_id_from_path
import matplotlib.pyplot as plt
import matplotlib
import argparse
from tensorflow.keras.callbacks import *
import data_preparation
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("--train", type=int, help="Provide number of training epochs", required=False)
    parser.add_argument("--load-weights", type=str, help="Path to saved weights", required=False)
    parser.add_argument("--generate-tfrecord", help="Generate tfrecord file", required=False, action='store_true')
    parser.add_argument("--tfrecord-path", help="Path to tfrecord file", required=False)
    parser.add_argument("--augment-data", help="Do augment data", required=False, action="store_true")
    parser.add_argument("--validation-split", type=float, help="Validation split of training data", default=0.05)
    parser.add_argument("--learning-rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=128)
    parser.add_argument("--model-data-dir", type=str, help="Path to store saved models", default="./models/")
    parser.add_argument("--max-size", type=int, help="max size", required=False)

    # Evaluation options
    parser.add_argument("--evaluate", type=str, help="Do evaluate model on images from given directory",
                        required=False)

    args = parser.parse_args()

    shapes = {"input": (128, 128, 3), "output": (62, )}
    train_image_path = "./data/"
    train_image_labels_path = "./data/labels.csv"

    # Create the generator
    train_generator = DataGenerator(
        train_image_path,
        train_image_labels_path,
        data_shapes=shapes,
        batch_size=args.batch_size,
        is_prepared=False if args.generate_tfrecord else True,
        augment_data=args.augment_data,
        validation_split=args.validation_split,
        tf_record_file_path=args.tfrecord_path if args.tfrecord_path else "",
        max_size=args.max_size if args.max_size else ""
    )

    # Exit if no more actions are specified
    if not(args.train or args.evaluate):
        dbg("Nothing more to do, exiting")
        exit()

    # check trailing slash
    model_dir = args.model_data_dir
    if model_dir[-1] != "/":
        model_dir += "/"

    # Build the model
    model_util = ModelCreation(
        input_shape=shapes["input"],
        output_shape=shapes["output"],
        learning_rate=args.learning_rate,
        model_data_dir=model_dir,
        saved_weights=args.load_weights  # when this parameter is not provided it will be False
    )
    rlop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    checkpoint = ModelCheckpoint('./models/auto-best.hdf5', monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')

    if args.train:
        # Train the model
        dbg("Training the model")
        model_util.model.fit_generator(
            train_generator.fak_you_generator(),
            steps_per_epoch=train_generator.get_steps_per_epoch(),
            epochs=args.train,
            validation_data=train_generator.fak_you_generator(validation=True),
            validation_steps=train_generator.get_validation_steps_per_epoch(),
            callbacks=[checkpoint]
        )
        # Save after training
        model_util.save_model()

    if args.evaluate:
        dbg("Evaluationg the model")
        images = data_preparation.get_validation_data(args.evaluate, '')
        results = model_util.evaluate(images)
        unparsed = data_preparation.unparse_labels(results, args.evaluate)
