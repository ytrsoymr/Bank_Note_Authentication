# Bank Note Authentication

## Overview
This project involves authenticating banknotes using machine learning techniques. The dataset consists of images captured from genuine and forged banknote-like specimens. An industrial camera, typically used for print inspection, was employed to capture the images. The images were digitized with a resolution of about 660 dpi and are of size 400x400 pixels. Wavelet Transform tools were used to extract features from the images.

## Dataset Link
[Bank Note Authentication Dataset](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

## Features
The dataset contains the following features:
- **variance**: Variance of Wavelet Transformed image.
- **skewness**: Skewness of Wavelet Transformed image.
- **curtosis**: Curtosis of Wavelet Transformed image.
- **entropy**: Entropy of image.
- **class**: 0 for genuine and 1 for forged.

## Requirements
- Python 3.x ,
- pandas ,
- numpy ,
- scikit-learn ,
- pickle

## Conclusion
This project demonstrates the use of machine learning for banknote authentication. The Random Forest Classifier achieved a high accuracy of 99.03%, indicating that the model can effectively differentiate between genuine and forged banknotes based on extracted image features.
