# Social Media and Mental Health Balance
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

## Project Overview
This project analyzes the relationship between social media usage, lifestyle habits, and mental well-being. Using a dataset of 500 observations, we developed and compared two Artificial Neural Network (ANN) models to predict a user's "Happiness Index".

The goal was to classify users into "High Happiness" or "Low/Average Happiness" based on features such as daily screen time, sleep quality, stress levels, and exercise frequency.

### Dataset
The dataset relates social media usage to mental health.
* **Source:** [Social Media and Mental Health Balance (Kaggle)](https://www.kaggle.com/datasets/prince7489/mental-health-and-social-media-balance-dataset/data)
* **Observations:** 500
* **Target Variable:** `Happiness_Index` (Binarized: >8 is High Happiness, <=8 is Low/Average).

### Key Features
* **Numerical:** Age, Daily Screen Time, Sleep Quality, Stress Level, Exercise Frequency.
* **Categorical:** Gender, Social Media Platform.


## Technology Stack
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib** (Visualization)
* **Scikit-Learn** (Preprocessing & Metrics)
* **TensorFlow / Keras** (Neural Networks)
* **MLflow** (Experiment Tracking & Model Registry)

## Architecture & Pipeline

### 1. High-Level ML Pipeline
The project follows a standard End-to-End Machine Learning workflow, integrated with **MLflow** for experiment tracking.

```mermaid
graph LR
    A[Raw Dataset] --> B(Data Preprocessing)
    B --> C{Split Data}
    C -->|Train Set| D[Model Training]
    C -->|Test Set| E[Evaluation]
    D --> F[MLflow Tracking]
    F -->|Log Metrics| G((Model Registry))
    E --> H[Final Predictions]
    
    subgraph Preprocessing
    B1[One-Hot Encoding]
    B2[StandardScaler]
    B3[Binarization]
    end
