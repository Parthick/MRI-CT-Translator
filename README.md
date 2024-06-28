# MRI-CT Image Translator
![Background](Designer(2).jpeg)
This project aims to translate MRI images to CT images and vice versa using a CycleGAN architecture. The web application is built with Django and provides an interface for users to upload MRI or CT images and receive the translated output. The project leverages TensorFlow, Keras, DVC, MLflow, and Google Drive for model management and deployment.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [DVC and MLflow Integration](#dvc-and-mlflow-integration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses CycleGAN to perform image-to-image translation between MRI and CT scans. CycleGAN is a type of Generative Adversarial Network (GAN) that learns to translate images from one domain to another without requiring paired examples.

## Architecture

The project architecture consists of:
- **CycleGAN Model**: Trained using TensorFlow and Keras.
- **Django Web Application**: Provides a user-friendly interface for uploading and translating images.
- **DVC**: Manages datasets and model versions.
- **MLflow**: Tracks experiments and model versions.
- **Google Drive**: Stores datasets and models.

## Features

- **Image Translation**: Translate MRI images to CT images and vice versa.
- **Web Interface**: Upload images and view results directly from the browser.
- **Model Management**: Track and version models using DVC and MLflow.
- **Cloud Storage**: Store datasets and models on Google Drive.

## Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Keras
- Django 3.2+
- DVC
- MLflow
- Google Drive API

## Installation

### Clone the repository

```bash
git clone https://github.com/Towet-Tum/MRI-CT-Translator.git
cd MRI-CT-Translater
```

###  Set up a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```
###  Install dependencies
```bash
pip install -r requirements.txt
```
###  Set up DVC

```bash
dvc init
dvc remote add -d gdrive remote-url
```

### Set up Django

```bash
cd Translator
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

# Usage

1 Start the Django server:

```bash

python manage.py runserver
```
2 Access the web application:
Open your browser and navigate to http://127.0.0.1:8000.

3 Upload an image:
Use the interface to upload an MRI or CT image and receive the translated result.

# Model Training
Dataset Preparation

Ensure your MRI and CT datasets are structured and stored in your DVC remote storage.
Training the CycleGAN Model

Configure your training script:
Update the dataset paths and model parameters in params.yaml.

Run the training script:
    ```bash 
    python main.py
    or dvc repro
    ```

4 Push the trained model to DVC:
```bash 
    dvc add models/
    dvc push
```

# DVC and MLflow Integration
## Track Experiments

1 Set up MLflow tracking:
Configure your MLflow tracking server and update the tracking URI in your evaluation script.

2 Log experiments:
Use MLflow to log parameters, metrics, and models during model evaluation.
# Version Control

1 Commit DVC changes:
```bash 
    dvc add data/processed
    git add data/processed.dvc
    git commit -m "Add processed data"
```
2 Push changes:
```bash 
    dvc push
    git push
```



