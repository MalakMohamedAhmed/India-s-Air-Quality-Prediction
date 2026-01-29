import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5")
scaler_x = joblib.load("scaler_x.sav")
scaler_y = joblib.load("scaler_y.sav")


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 😊"
    elif aqi <= 100:
        return "Moderate 🙂"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 😷"
    elif aqi <= 200:
        return "Unhealthy 🤒"
    elif aqi <= 300:
        return "Very Unhealthy 🥵"
    else:
        return "Hazardous ☠️"


st.title("🌍 AQI Prediction App using ANN")
st.write("Enter air pollution values to predict Air Quality Index (AQI)")


pm25 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, co, no2, so2, o3]])
    input_scaled = scaler_x.transform(input_data)

    y_pred_scaled = model.predict(input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    aqi_value = float(y_pred[0][0])
    category = get_aqi_category(aqi_value)

    st.success(f"✅ Predicted AQI: {aqi_value:.2f}")
    st.info(f"📊 AQI Category: {category}")

