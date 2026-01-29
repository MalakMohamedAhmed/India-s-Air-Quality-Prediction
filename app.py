import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os

# --- 1. Load Assets Safely ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.h5")
    # Use .sav files as seen in your Hugging Face screenshot
    scaler_x = joblib.load("scaler_x.sav")
    scaler_y = joblib.load("scaler_y.sav")
    return model, scaler_x, scaler_y

try:
    model, scaler_x, scaler_y = load_assets()
except Exception as e:
    st.error(f"Error loading model/scalers: {e}")
    st.stop()

# --- 2. Helper Functions ---
def get_aqi_category(aqi):
    if aqi <= 50: return "Good 😊"
    elif aqi <= 100: return "Moderate 🙂"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups 😷"
    elif aqi <= 200: return "Unhealthy 🤒"
    elif aqi <= 300: return "Very Unhealthy 🥵"
    else: return "Hazardous ☠️"

# --- 3. UI Layout ---
st.title("🌍 AQI Prediction App using ANN")
st.write("Enter pollutant concentrations to predict the Air Quality Index (AQI)")

col1, col2 = st.columns(2)

with col1:
    so2 = st.number_input("SO2 (Sulfur Dioxide)", min_value=0.0)
    no2 = st.number_input("NO2 (Nitrogen Dioxide)", min_value=0.0)
    pm10 = st.number_input("PM10 (Coarse Particulate)", min_value=0.0)

with col2:
    pm25 = st.number_input("PM2.5 (Fine Particulate)", min_value=0.0)
    co = st.number_input("CO (Carbon Monoxide)", min_value=0.0)
    o3 = st.number_input("O3 (Ozone)", min_value=0.0)

# --- 4. Prediction Logic ---
if st.button("Predict AQI"):
    # Ensure order matches your training dataset columns exactly!
    # Example order: [so2, no2, rspm, spm, pm2_5] - Adjust to match your model
    input_data = np.array([[so2, no2, pm10, pm25, co, o3]]) 
    
    try:
        input_scaled = scaler_x.transform(input_data)
        y_pred_scaled = model.predict(input_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        aqi_value = float(y_pred[0][0])
        category = get_aqi_category(aqi_value)
        
        st.success(f"✅ Predicted AQI: {aqi_value:.2f}")
        st.info(f"📊 AQI Category: {category}")
    except Exception as e:
        st.error(f"Prediction Error: {e}. Check if the number of inputs matches your model's training features.")
