# MaskML - Face Mask Detection System

A machine learning project that detects whether people are wearing face masks correctly, incorrectly, or not at all using computer vision and classical ML techniques.

## Project Overview

This system uses Support Vector Machine (SVM) with Principal Component Analysis (PCA) to classify face mask usage into three categories:
- **with_mask** - Properly worn face mask
- **mask_weared_incorrect** - Incorrectly worn face mask  
- **without_mask** - No face mask detected

## Features

- **Classical ML Pipeline**: SVM classifier with advanced preprocessing
- **Real-time Detection**: Live webcam face mask detection
- **Face Detection**: Automatic face detection using Haar cascades
- **Confidence Scores**: Probability-based predictions
- **Easy Training**: Simple dataset training process

## Project Structure
MaskML/
├── train.py # Model training and single image prediction 
├── app.py # Real-time webcam detection 
├── dataset/ # Training images organized by class 
│ ├── with_mask/ 
│ ├── without_mask/ 
│ └── mask_weared_incorrect/ 
└── mask_detector_model.pkl # Trained model (generated after training)


## 🛠️ Installation

1. **Clone the repository:**

  git clone <https://github.com/Vedaant-VBD/Mask-Detection-ML>


2. **Install required dependencies:**

  pip install opencv-python numpy scikit-learn pickle-mixin
