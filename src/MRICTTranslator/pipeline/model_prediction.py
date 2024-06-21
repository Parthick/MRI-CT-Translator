import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from MRICTTranslator.utils.common import InstanceNormalization


class PredictionPipeline:
    def __init__(self, filename: str):
        self.filename = filename
        self.model_AtoB_path = "/home/towet/Desktop/OpenProjects/MRI-CT-Translator/artifacts/training/tf_g_model_AtoB_001742.h5"
        self.model_BtoA_path = "/home/towet/Desktop/OpenProjects/MRI-CT-Translator/artifacts/training/tf_g_model_BtoA_001742.h5"
        self.model_AtoB = self.load_model(self.model_AtoB_path)
        self.model_BtoA = self.load_model(self.model_BtoA_path)

    @staticmethod
    def load_model(path: str) -> tf.keras.Model:
        custom_objects = {
            "InstanceNormalization": InstanceNormalization,
        }
        return tf.keras.models.load_model(path, custom_objects=custom_objects)

    def process_image(self) -> np.ndarray:
        test_image = load_img(self.filename)
        test_image = img_to_array(test_image)
        test_image = tf.image.resize(test_image, (256, 256))
        test_image = (test_image - 127.5) / 127.5  # Normalize to [-1, 1]
        return np.expand_dims(
            test_image, axis=0
        )  # Convert single image to batch format

    def denormalize_image(self, image):
        # Convert from [-1, 1] to [0, 255]
        image = (image * 127.5) + 127.5
        return np.uint8(image)

    def predict_mri_to_ct(self):
        mri_image = self.process_image()
        ct_generated = self.model_BtoA.predict(mri_image)
        mri_reconstructed = self.model_AtoB.predict(ct_generated)
        return mri_image, ct_generated, mri_reconstructed

    def predict_ct_to_mri(self):
        ct_image = self.process_image()
        mri_generated = self.model_AtoB.predict(ct_image)
        ct_reconstructed = self.model_BtoA.predict(mri_generated)
        return ct_image, mri_generated, ct_reconstructed
