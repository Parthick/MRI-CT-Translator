# mritoct
import tensorflow as tf
from MRICTTranslator.utils.common import (
    preprocess_data,
    load_images,
    generate_fake_samples,
    generate_real_samples,
    summarize_performance,
    save_models,
    update_image_pool,
)
from MRICTTranslator.models.cycle_gan import *
import numpy as np
from datetime import datetime

from MRICTTranslator.models.cycle_gan import *
from MRICTTranslator.entity.config_entity import ModelTrainingConfig


class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train(
        self,
        d_model_A,
        d_model_B,
        g_model_AtoB,
        g_model_BtoA,
        c_model_AtoB,
        c_model_BtoA,
        dataset,
        epochs=1,
    ):
        # define properties of the training run
        (
            n_epochs,
            n_batch,
        ) = (
            epochs,
            1,
        )  # batch size fixed to 1 as suggested in the paper
        # determine the output square shape of the discriminator
        n_patch = d_model_A.output_shape[1]
        # unpack dataset
        trainA, trainB = dataset
        # prepare image pool for fake images
        poolA, poolB = list(), list()
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs

        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples from each domain (A and B)
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
            # generate a batch of fake samples using both B to A and A to B generators.
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            # update fake images in the pool. Remember that the paper suggstes a buffer of 50 images
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)

            # update generator B->A via the composite model
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
                [X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA]
            )
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

            # update generator A->B via the composite model
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
                [X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB]
            )
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

            # summarize performance
            # Since our batch size =1, the number of iterations would be same as the size of our dataset.
            # In one epoch you'd have iterations equal to the number of images.
            # If you have 100 images then 1 epoch would be 100 iterations
            print(
                "Iteration>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]"
                % (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
            )
            # evaluate the model performance periodically
            # If batch size (total images)=100, performance will be summarized after every 75th iteration.
            if (i + 1) % (bat_per_epo * 1) == 0:
                # plot A->B translation
                summarize_performance(i, g_model_AtoB, trainA, "AtoB")
                # plot B->A translation
                summarize_performance(i, g_model_BtoA, trainB, "BtoA")
            if (i + 1) % (bat_per_epo * 5) == 0:
                # save the models
                # #If batch size (total images)=100, model will be saved after
                # every 75th iteration x 5 = 375 iterations.
                save_models(
                    i,
                    g_model_AtoB,
                    g_model_BtoA,
                    self.config.model_path_AtoB,
                    self.config.model_path_BtoA,
                )

    def model_train(self):
        # dataA is the CT scans and dataB is the MRI scans
        dataA = load_images(self.config.dataset + "trainA/")
        dataB = load_images(self.config.dataset + "trainB/")
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
