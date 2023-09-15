# Satellite-Ground Cover CLIP Model

This project aims to build a Contrastive Language-Image Pretraining (CLIP) model for satellite images and their corresponding ground cover classes.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [How to Use](#how-to-use)
4. [Model Architecture](#model-architecture)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)


## Introduction

The project focuses on leveraging CLIP to associate features in satellite images with their corresponding ground cover classes. 

<!-- Add more details, perhaps some use cases or issues that this project aims to solve. -->

## Dependencies

- Python 3.x
- PyTorch
- rasterio
- PIL
- torchvision

<!-- Include how to install them or provide a `requirements.txt` file -->

## How to Use

Instructions for how to run the code.

<!-- Add more detailed steps to run the project -->
## Model Architecture

Our GeoCLIP model architecture consists of two main components: the satellite image encoder and the ground cover class encoder. They are trained in a contrastive manner using a specialized loss function.

### Architecture Diagram
<!-- Add the architecture diagram below this line -->
<img src="./docs/imgs/geoclip_v1.png" alt="GeoCLIP Architecture" width="1200"/>

<!-- If you have more detailed description or parameters, list them here -->


## Data Preparation

### Satellite Data

Explain how and where to get the satellite images, and how they need to be formatted or preprocessed for the project.

<!-- Add more details on data format, naming conventions, folder structure, etc. -->

### Ground Cover Class Data

Details on how to prepare the ground cover class data.

<!-- Add more details on data format, naming conventions, folder structure, etc. -->

## Training

Details on how to train the model, including any special parameters that should be set.

<!-- Code snippets for running the training script, setting different flags, etc. -->

## Results

<!-- Add details about the results you obtained, any metrics you're tracking, etc. -->
<!-- Include charts or images if applicable -->

## Acknowledgements

<!-- Any acknowledgements, citations, references, etc. -->

<!-- End of README.md -->
