import os
from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings
from .models import ImagePrediction
from MRICTTranslator.pipeline.model_prediction import PredictionPipeline
from tensorflow.keras.preprocessing.image import array_to_img


def predict_and_store(request):
    if request.method == "POST":
        if "image" not in request.FILES:
            return render(request, "upload.html", {"error": "No image uploaded"})

        uploaded_image = request.FILES["image"]
        translation_type = request.POST.get("translation_type")

        # Save the uploaded file using Django's file storage system
        fs_upload = FileSystemStorage(
            location=os.path.join(settings.MEDIA_ROOT, "uploads")
        )
        filename = fs_upload.save(uploaded_image.name, uploaded_image)
        image_path = fs_upload.path(filename)

        # Create the prediction pipeline instance
        pipeline = PredictionPipeline(filename=image_path)

        # Predict the images
        if translation_type == "mri_to_ct":
            original_image, generated_image, reconstructed_image = (
                pipeline.predict_mri_to_ct()
            )
            original_field = "original_mri"
            generated_field = "generated_ct"
            reconstructed_field = "reconstructed_mri"
        else:
            original_image, generated_image, reconstructed_image = (
                pipeline.predict_ct_to_mri()
            )
            original_field = "original_ct"
            generated_field = "generated_mri"
            reconstructed_field = "reconstructed_ct"

        # Save images to disk using FileSystemStorage
        fs_save = FileSystemStorage(
            location=os.path.join(settings.MEDIA_ROOT, "predictions")
        )

        def save_image(image_array, filename):
            image = array_to_img(image_array[0])
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return fs_save.save(filename, ContentFile(buffer.getvalue()))

        original_image_path = save_image(original_image, f"{original_field}.jpg")
        generated_image_path = save_image(generated_image, f"{generated_field}.jpg")
        reconstructed_image_path = save_image(
            reconstructed_image, f"{reconstructed_field}.jpg"
        )

        # Save the prediction details to the database
        prediction = ImagePrediction(
            original_mri=(
                os.path.join("predictions", os.path.basename(original_image_path))
                if translation_type == "mri_to_ct"
                else None
            ),
            generated_ct=(
                os.path.join("predictions", os.path.basename(generated_image_path))
                if translation_type == "mri_to_ct"
                else None
            ),
            reconstructed_mri=(
                os.path.join("predictions", os.path.basename(reconstructed_image_path))
                if translation_type == "mri_to_ct"
                else None
            ),
            original_ct=(
                os.path.join("predictions", os.path.basename(original_image_path))
                if translation_type == "ct_to_mri"
                else None
            ),
            generated_mri=(
                os.path.join("predictions", os.path.basename(generated_image_path))
                if translation_type == "ct_to_mri"
                else None
            ),
            reconstructed_ct=(
                os.path.join("predictions", os.path.basename(reconstructed_image_path))
                if translation_type == "ct_to_mri"
                else None
            ),
        )
        prediction.save()

        return render(request, "prediction_result.html", {"prediction": prediction})

    return render(request, "upload.html")
