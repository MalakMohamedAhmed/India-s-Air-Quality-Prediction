# Delhi-NCR Air Quality Index: A Comparative ML & DL Approach
  This project provides a comprehensive analysis and prediction of the Air Quality Index (AQI) in Delhi-NCR. It evaluates the performance of Classical Machine       Learning algorithms against Deep Learning (ANN) architectures to identify the most effective method for environmental forecasting.

# **🚀 Project Overview**
  Predicting AQI is challenging due to extreme seasonal variance and high feature dependency. This project implements a full data science pipeline, achieving a peak $R^2$ score of 99.31%.

  **Key Highlights**
    - Dual-Model Approach:
    - Compares traditional regression models with multi-layer Artificial Neural Networks.
    - Large-Scale Data: Processed 201,664 observations from 23 monitoring stations.
    - High Precision: Achieved an RMSE of 0.08, ensuring reliability even during high-pollution "Hazardous" events.

# **🛠️ Methodology & Tech Stack1.**
  1. Classical Machine Learning
    We utilized ensemble methods and linear models to establish a baseline:
    - Models: Random Forest Regressor, Plynomial, or Linear Regression.
    - Focus: Feature importance and interpretability.
  2. Deep Learning (ANN)
    A custom-built Neural Network was developed to capture complex, non-linear atmospheric patterns:
    - Framework: TensorFlow / Keras.Architecture: Multi-layer Sequential model with ReLU activation and Adam optimization.

# **🧪 Advanced Feature Engineering**
  To solve the high Multicollinearity (VIF) issues typical in pollution data, we implemented:
  - Gaseous Pollutant Index (GPI): Merged $SO_2$, $CO$, and $NO_2$ into a single weighted feature.
  - Particulate Analysis: Isolated "Coarse Fraction" ($PM_{10} - PM_{2.5}$) to distinguish between dust and combustion particles.
  - Weather Integration: Integrated Temperature, Humidity, and Wind Speed to model pollutant dispersion.
