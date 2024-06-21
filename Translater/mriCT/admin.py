from django.contrib import admin
from .models import ImagePrediction


# Register your models here.
class ImagePredictionAdmin(admin.ModelAdmin):
    pass
    # list_display = ["original_mri", "generated_ct", "reconstructed_mri", "created_at"]


admin.site.register(ImagePrediction)
