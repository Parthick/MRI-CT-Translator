# models.py
from django.db import models


class ImagePrediction(models.Model):
    original_mri = models.ImageField(upload_to="predictions/", null=True, blank=True)
    generated_ct = models.ImageField(upload_to="predictions/", null=True, blank=True)
    reconstructed_mri = models.ImageField(
        upload_to="predictions/", null=True, blank=True
    )
    original_ct = models.ImageField(upload_to="predictions/", null=True, blank=True)
    generated_mri = models.ImageField(upload_to="predictions/", null=True, blank=True)
    reconstructed_ct = models.ImageField(
        upload_to="predictions/", null=True, blank=True
    )

    def __str__(self):
        return f"Prediction {self.id}"
