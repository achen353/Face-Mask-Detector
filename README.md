
<h1 align="center">Proper Mask Wearing Detection and Alarm System</h1>

<div align="center">
    <strong>A lightweight face mask detection that is easy to deploy</strong>
</div>

<div align="center">
    Trained on Tensorflow/Keras. Deployed using Dash on Google App Engine. 
</div>

<br/>

<div align="center">
    <!-- Python version -->
    <img src="https://img.shields.io/badge/python-v3.8-blue?style=flat-square"/>
    <!-- Last commit -->
    <img src="https://img.shields.io/github/last-commit/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Stars -->
    <img src="https://img.shields.io/github/stars/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Forks -->
    <img src="https://img.shields.io/github/forks/achen353/Face-Mask-Detector?style=flat-square"/>
    <!-- Open Issues -->
    <img src="https://img.shields.io/github/issues/achen353/Face-Mask-Detector?style=flat-square"/>
</div>

<br/>

<div align="center">
    <img src="./readme_assets/readme_cover.png"/>
</div>

<br/>

*Read this in [繁體中文](README.zh-tw.md).*

## Table of Contents
- [Features](#features)
- [About](#about)
- [Frameworks and Libraries](#frameworkslibraries)
- [Datasets](#datasets)
- [Training Results](#training-results)
- [Requirements](#requirements)
- [Setup](#setup) 
- [How to Run](#how-to-run)
- [Dash App Demo](#dash-app-demo)
- [Credits](#credits)
- [License](#license)

## Features
- __Lightweight models:__  only `2,422,339` and `2,422,210` parameters for the MFN and RMFD models, respectively
- __Detection of multiple faces:__ able to detect multiple faces in one frame
- __Support for detection in webcam stream:__ our app supports detection in images and video streams 
- __Support for detection of improper mask wearing:__ our MFN model is able to detect improper mask wearing including
  (1) uncovered chin, (2) uncovered nose, and (3) uncovered nose and mouth.

## About
This app detects human faces and proper mask wearing in images and webcam streams. 

Under the COVID-19 pandemic, wearing
mask has shown to be an effective means to control the spread of virus. The demand for an effective mask detection on 
embedded systems of limited computing capabilities has surged, especially in highly populated areas such as public 
transportations, hospitals, etc. Trained on MobileNetV2, a state-of-the-art lightweight deep learning model on 
image classification, the app is computationally efficient to deploy to help control the spread of the disease.

While many work on face mask detection has been developed since the start of the pandemic, few distinguishes whether a
mask is worn correctly or incorrectly. Given the discovery of the new coronavirus variant in UK, we aim to provide a 
more precise detection model to help strengthen enforcement of mask mandate around the world.

## Frameworks and Libraries
- __[OpenCV](https://opencv.org/):__ computer vision library used to process images
- __[OpenCV DNN Face Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py):__ 
  Caffe-based Single Shot-Multibox Detector (SSD) model used to detect faces
- __[Tensorflow](https://www.tensorflow.org/) / [Keras](https://keras.io/):__ deep learning framework used to build and train our models
- __[MobileNet V2](https://arxiv.org/abs/1801.04381):__ lightweight pre-trained model available in Keras Applications; 
  used as a base model for our transfer learning
- __[Dash](https://plotly.com/dash/):__ framework built upon Plotly.js, React and Flask; used built the demo app

## Datasets
We provide two models trained on two different datasets. 
Our RMFD dataset is built from the [Real World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) 
and the MFN dataset is built from the [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) and 
[Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset).

### RMFD dataset
This dataset consists of __4,408__ images:
- `face_no_mask`: 2,204 images
- `face_with_mask`: 2,204 images

Each image is a cropped real-world face image of unfixed sizes. The `face_no_mask` data is randomly sampled from the 90,568 no mask
data from the Real World Masked Face Dataset and the `face_with_mask` data entirely provided by the original dataset.

### MFN dataset
This dataset consists of __200,627__ images:
- `face_with_mask_correctly`: 67,193 images
- `face_with_mask_incorrectly`: 66,899 images
- `face_no_mask`: 66,535 images

The `face_with_mask_correctly` and `face_with_mask_incorrectly` classes consist of the resized 128*128 images from 
the original MaskedFace-Net work without any sampling. The `face_no_mask` is built from the 
Flickr-Faces-HQ Dataset (FFHQ) upon which the MaskedFace-Net data was created.
All images in MaskedFace-Net are morphed mask-wearing images and `face_with_mask_incorrectly` consists of 10% uncovered chin, 10% uncovered nose, and 80% uncovered nose and mouth images.

### Download
The dataset is now available [here](https://drive.google.com/file/d/1Y1Y67osv8UBKn_ANckCXPvY2aZqv1Cha/view?usp=sharing)! (June 11, 2021)

## Training Results
Both models are trained on 80% of their respectively dataset and validated/tested on the other 20%. They both achieved 99%
accuracy on their validation data.

MFN Model                             |  RMFD Model
:------------------------------------:|:--------------------------------------:
![](./figures/train_plot_MFN.jpg)   |  ![](./figures/train_plot_RMFD.jpg) 


However, the MFN model sometimes classifies `face_no_mask` as `face_with_mask_incorrectly`. Though this would not affect
goal of reminding people to wear mask properly, any suggestion to improve the model is welcomed.

## Requirements
This project is built using Python 3.8 on MacOS Big Sur 11.1. The training of the model is performed on custom GCP 
Compute Engine (8 vCPUs, 13.75 GB memory) with `tensorflow==2.4.0`. All dependencies and packages are listed in
`requirements.txt`. 

Note: We used `opencv-python-headless==4.5.1` due to an [issue](https://github.com/skvark/opencv-python/issues/423) 
with `cv2.imshow` on MacOS Big Sur. However, recent release of `opencv-python 4.5.1.48` seems to have fixed the problem.
Feel free to modify the `requirements.txt` before you install all the listed packages.

## Setup
1. Open your terminal, `cd` into where you'd like to clone this project, and clone the project:
```
$ git clone https://github.com/achen353/Face-Mask-Detector.git
```
2. Download and install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html).
3. Create an environment with the packages on `requirements.txt` installed:
```
$ conda create --name env_name --file requirements.txt
```
4. Now you can `cd` into the clone repository to run or inspect the code.

## How to Run

### To detect masked faces in images
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_images.py -i <image-path> [-m <model>] [-c <confidence>]
```

### To detect masked faces in webcam streams
`cd` into `/src/` and enter the following command:
```
$ python detect_mask_video.py [-m <model>] [-c <confidence>]
```

### To train the model again on the dataset
`cd` into `/src/` and enter the following command:
```
$ python train.py [-d <dataset>]
```
Make sure to modify the paths in `train.py` to avoid overwriting existing models.

Note: 
- `<image-path>` should be relative to the project root directory instead of `/src/`
- `<model>` should be of `str` type; accepted values are `MFN` and `RMFD` with default value `MFN`
- `<confidence>` should be `float`; accepting values between `0` and `1` with default value `0.5`
- `<dataset>` should be of `str` type; accepted values are `MFN` and `RMFD` with default value `MFN`

## Dash App Demo
The demo of the app is available [here](https://face-mask-detection-300106.wl.r.appspot.com); it is still under testing.

### Run the app yourself
1. Modify `app.run_server(host='0.0.0.0', port=8080, debug=True)` to `app.run_server(debug=True)`:
2. Run the app:
```
$ python main.py
```
3. Enter `http://127.0.0.1:8050/` in your browser to open the app on the Dash app's default host and port. Feel free to modify
the host and port number if the default port is taken.

## Credits
- 口罩遮挡人脸数据集（[Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) ，RMFD）
- Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of 
  correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, 
  Elsevier, 2020, [DOI:10.1016/j.smhl.2020.100144](https://doi.org/10.1016/j.smhl.2020.100144)
- Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection 
  mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", 
  CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 
  2020, [DOI:10.32604/cmes.2020.011663](DOI:10.32604/cmes.2020.011663)
- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
- [Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Object Detection](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-object-detection)

## License
[MIT © Andrew Chen](https://github.com/achen353/Face-Mask-Detector/blob/master/LICENSE)
