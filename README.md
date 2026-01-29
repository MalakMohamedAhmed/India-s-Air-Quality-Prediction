---
title: Indian Air Quality Prediction
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
  - streamlit
pinned: false
short_description: ANN model predicts india's air quality
---

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/vishardmehta/delhi-pollution-aqi-dataset)
# 🌍 India Air Quality Index Analysis & Prediction
An end-to-end Data Science project focusing on predicting the Air Quality Index (AQI) across various Indian states using **Classical Machine Learning** and **Deep Learning (ANN)**.

---

## 📌 Table of Contents
* [Project Overview](#project-overview)
* [Dataset](#Dataset)
* [Data Engineering](#data-engineering)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Modeling & Results](#modeling--results)
* [Technical Implementation](#technical-implementation)

---

## 🚀 Project Overview
Air pollution is a critical environmental issue in India. This project analyzes historical data (200,000+ records) to build a predictive system that can forecast AQI based on chemical pollutants ($SO_2$, $NO_2$, $PM_{2.5}$, etc.).

**Core Objective:** To compare the efficacy of traditional regression models against Artificial Neural Networks (ANN) in predicting high-variance pollution spikes.

---

## 📂 Dataset
The dataset used in this project is sourced from Kaggle's **India Air Quality Data**. It contains historical daily air quality data across various cities in India.

* **Source:** [Delhi Pollution AQI Dataset on Kaggle](https://www.kaggle.com/datasets/vishardmehta/delhi-pollution-aqi-dataset)
* **Size:** 200,000+ observations
* **Format:** CSV

---

## 📊 Exploratory Data Analysis
Detailed EDA was conducted to provide environmental context to the numbers:

### Key Insights:
* **The Winter Peak:** Pollution levels spike significantly in winter due to **Thermal Inversion**, which traps particulate matter near the ground.
* **Vertical Mixing:** A consistent dip in AQI is observed around midday. This occurs because solar heating increases the **mixing height**, allowing pollutants to disperse vertically.
* **Industrial Hotspots:** Geographic mapping identified states like Delhi, West Bengal, and Jharkhand as high-risk zones for $SO_2$ and $NO_2$ concentrations.

---

## 🛠️ Data Engineering
Given the high complexity of the dataset, several preprocessing steps were vital:

### 1. Feature Engineering (The Indices)
Following Indian government standards, individual sub-indices were calculated for:
* **si** ($SO_2$ Index)
* **ni** ($NO_2$ Index)
* **rpi** (Respirable Particulate Index)
* **spi** (Suspended Particulate Index)

### 2. Handling Multicollinearity (VIF)
We used **Variance Inflation Factor (VIF)** to identify redundant features. High-VIF variables like $NO_2$ were managed through a custom **Gaseous Pollutant Index (GPI)** to ensure model stability and prevent overfitting.

---

## 🧠 Modeling & Results
The project implements a dual-modeling approach:

### 1. Classical Machine Learning
Baseline models were established using **Linear Regression** and **Ensemble Methods** (Random Forest/Gradient Boosting) to interpret feature importance.

### 2. Deep Learning (ANN)
A multi-layer **Artificial Neural Network** was built using TensorFlow/Keras.
* **Optimizer:** Adam
* **Activation:** ReLU (Hidden Layers), Linear (Output)
* **Epochs:** 40 (with early stopping checkpoints)

### 📈 Performance Summary
| Metric | Result |
| :--- | :--- |
| **R² Score** | **99.31%** |
| **RMSE** | **0.081** |
| **MAE** | **0.013** |

---
## 🚀 Deployment & Presentation

### 💻 Live Streamlit Application
The model has been deployed as an interactive web application using **Streamlit** and is hosted on **Hugging Face Spaces**. 
* **Real-time Prediction:** Users can input pollutant concentrations ($SO_2, NO_2, PM_{2.5}$, etc.) and weather parameters to get an instant AQI forecast.
* **Accessibility:** This deployment demonstrates the transition of the model from a research environment to a user-friendly production tool.

👉 **[Access the Live App on Hugging Face Spaces](INSERT_YOUR_HUGGING_FACE_LINK_HERE)**

---

### 📊 Presentation & Documentation
To facilitate the communication of findings to stakeholders, a comprehensive set of presentation slides was developed.
* **Visual Data Storytelling:** Detailed slides covering the problem statement, data sources, and the environmental "why" behind the insights.
* **Architecture Walkthrough:** Visual representations of the ANN structure and the feature engineering pipeline.
* **Impact Analysis:** Clear visualizations of model performance and how it can be utilized for public health alerts.

📂 **[Download/View Presentation Slides](INSERT_LINK_TO_YOUR_SLIDES_HERE)**

---

## ⚙️ Technical Implementation

### The Whole python notebook
```bash
India's_Air_Quality_Prediction.ipynp
