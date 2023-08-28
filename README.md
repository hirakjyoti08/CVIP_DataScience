# CVIP_DataScience

# Breast Cancer Prediction Project

This project aims to predict breast cancer diagnosis using machine learning techniques. The dataset used in this project contains various attributes measured from breast cancer biopsies, and the goal is to predict whether a tumor is malignant (M) or benign (B).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Selection](#model-selection)
- [Performance Evaluation](#performance-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Deployment](#model-deployment)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Introduction

Breast cancer is a significant health concern, and early detection is crucial for effective treatment. This project uses machine learning algorithms to predict whether a tumor is malignant or benign based on attributes extracted from breast cancer biopsies.

## Dataset

The dataset used for this project contains information about various attributes of breast cancer biopsies. The dataset has been preprocessed to match the requirements of the machine learning algorithms.

## Data Preprocessing

- The 'diagnosis' column was transformed to binary values (0 for benign and 1 for malignant).
- The dataset was split into features (X) and target labels (Y).
- The data was further divided into training and test sets for model evaluation.

## Exploratory Data Analysis

- Data visualization techniques were employed to understand the distribution of attributes.
- Correlation matrices were visualized to explore relationships between attributes.

## Model Selection

Several machine learning algorithms were evaluated for their performance in predicting breast cancer diagnosis, including Decision Tree, Support Vector Machine (SVM), Naive Bayes, and K-Nearest Neighbors.

## Performance Evaluation

- Cross-validation was used to assess the performance of each model.
- The models were evaluated based on accuracy, precision, recall, and F1-score.

## Hyperparameter Tuning

The SVM model was selected for hyperparameter tuning using GridSearchCV to find the best combination of parameters that yield the highest accuracy.

## Model Deployment

The final SVM model with tuned hyperparameters was trained on the scaled training data and evaluated on the test dataset.

## Results

- The SVM model achieved an accuracy of approximately 95.6% on the test dataset.
- Precision, recall, and F1-score were also calculated, showing a well-rounded performance.

## How to Run

1. Install the required dependencies (see [Dependencies](#dependencies)).
2. Download the dataset and update the file path in the code.
3. Run the code to preprocess the data, explore it, train and evaluate different models, and deploy the final model.

## Dependencies

- Python (>=3.6)
- pandas
- numpy
- matplotlib
- scikit-learn

Install the required packages using the following command:

