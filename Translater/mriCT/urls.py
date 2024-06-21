# urls.py
from django.urls import path
from .views import predict_and_store

urlpatterns = [
    path("predict/", predict_and_store, name="predict_and_store"),
]
