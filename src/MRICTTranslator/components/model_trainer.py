from os import listdir
from numpy import asarray
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from MRICTTranslator.utils.common import preprocess_data, train
from MRICTTranslator.models.cycle_gan import *
from MRICTTranslator.entity.config_entity import ModelTrainingConfig


class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    # load all images in a directory into memory
    def load_images(self, path, size=(256, 256)):
        data_list = list()
        # enumerate filenames in directory, assume all are images
        for filename in listdir(path):
            # load and resize the image
            pixels = load_img(path + filename, target_size=size)
            # convert to numpy array
            pixels = img_to_array(pixels)
            # store
            data_list.append(pixels)
        return asarray(data_list)

    def model_train(self):
        # dataA is the CT scans and dataB is the MRI scans
        dataA = self.load_images(self.config.dataset + "trainA/")
        dataB = self.load_images(self.config.dataset + "trainB/")
        # load image data
        data = [dataA, dataB]
        dataset = preprocess_data(data)
        # define input shape based on the loaded dataset

        image_shape = dataset[0].shape[1:]
        # generator: A -> B
        g_model_AtoB = define_generator(image_shape)
        # generator: B -> A
        g_model_BtoA = define_generator(image_shape)
        # discriminator: A -> [real/fake]
        d_model_A = define_discriminator(image_shape)
        # discriminator: B -> [real/fake]
        d_model_B = define_discriminator(image_shape)
        # composite: A -> B -> [real/fake, A]
        c_model_AtoB = define_composite_model(
            g_model_AtoB, d_model_B, g_model_BtoA, image_shape
        )
        # composite: B -> A -> [real/fake, B]
        c_model_BtoA = define_composite_model(
            g_model_BtoA, d_model_A, g_model_AtoB, image_shape
        )
        start1 = datetime.now()
        # train models
        train(
            d_model_A,
            d_model_B,
            g_model_AtoB,
            g_model_BtoA,
            c_model_AtoB,
            c_model_BtoA,
            dataset,
            epochs=self.config.epochs,
        )

        stop1 = datetime.now()
        # Execution time of the model
        execution_time = stop1 - start1
        print("Execution time is: ", execution_time)
