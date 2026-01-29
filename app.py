import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# --- 1. Load Assets Safely ---
@st.cache_resource
def load_assets():
    # Use compile=False to avoid the 'mse' deserialization error
    model = tf.keras.models.load_model("model.h5", compile=False)
    scaler_x = joblib.load("scaler_x.sav")
    scaler_y = joblib.load("scaler_y.sav")
    return model, scaler_x, scaler_y

model, scaler_x, scaler_y = load_assets()

# --- 2. Feature Definitions ---
# Lists for dummy variables to be handled by dropdowns
stations = [
    'station_Bawana, Delhi', 'station_Dwarka Sec 8, Delhi', 'station_Faridabad New Town',
    'station_Faridabad Sec 16A', 'station_Ghaziabad Loni', 'station_Ghaziabad Vasundhara',
    'station_Greater Noida', 'station_Gurugram Sec 51', 'station_Gurugram Vikas Sadan',
    'station_ITO, Delhi', 'station_Jahangirpuri, Delhi', 'station_Mandir Marg, Delhi',
    'station_NSIT Dwarka, Delhi', 'station_Noida Sec 125', 'station_Noida Sec 62',
    'station_Okhla Phase 2, Delhi', 'station_Punjabi Bagh, Delhi', 'station_RK Puram, Delhi',
    'station_Rohini, Delhi', 'station_Shadipur, Delhi', 'station_Siri Fort, Delhi', 'station_Wazirpur, Delhi'
]

cities = ['city_Faridabad', 'city_Ghaziabad', 'city_Gurugram', 'city_Noida']

days_of_week = [
    'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday',
    'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
]

# --- 3. UI Layout ---
st.set_page_config(page_title="India AQI Predictor", layout="wide")
st.title("🌍 Indian Air Quality Prediction (ANN)")

with st.sidebar:
    st.header("📍 Location & Time")
    selected_city = st.selectbox("Select City", cities)
    selected_station = st.selectbox("Select Station", stations)
    selected_day = st.selectbox("Day of Week", ["Friday"] + days_of_week)  # Friday is the reference (all dummies 0)
    
    st.header("🗓️ Season")
    # Using checkboxes for seasons (bool, treated as 0.0 or 1.0)
    s_monsoon = st.checkbox("Post Monsoon")
    s_summer = st.checkbox("Summer")
    s_winter = st.checkbox("Winter")

st.subheader("🧪 Pollutant Concentrations & Weather")
col1, col2, col3 = st.columns(3)

with col1:
    month = st.slider("Month", 1, 12, 1)
    hour = st.slider("Hour", 0, 23, 12)
    is_weekend = st.radio("Is Weekend?", [0.0, 1.0])

with col2:
    o3 = st.number_input("O3", value=0.0, min_value=0.0)  # Prevent negative
    temp = st.number_input("Temperature", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0, min_value=0.0, max_value=100.0)
    wind_speed = st.number_input("Wind Speed", value=5.0, min_value=0.0)

with col3:
    visibility = st.number_input("Visibility", value=10.0, min_value=0.0)
    gpi = st.number_input("GPI (Gaseous Pollutant Index)", value=0.0, min_value=0.0)
    pm_coarse = st.number_input("PM Coarse", value=0.0, min_value=0.0)

# --- 4. Prediction Logic ---
if st.button("Predict Air Quality Index"):
    # Basic input validation
    if o3 < 0 or humidity < 0 or humidity > 100 or wind_speed < 0 or visibility < 0 or gpi < 0 or pm_coarse < 0:
        st.error("Please ensure all pollutant and weather values are non-negative and humidity is 0-100%.")
        st.stop()
    
    # Initialize input list with 45 features (excluding aqi and aqi_category)
    # Order: month, hour, is_weekend, o3, temperature, humidity, wind_speed, visibility, GPI, pm_coarse,
    # season_post_monsoon, season_summer, season_winter,
    # day_of_week_Monday, day_of_week_Saturday, day_of_week_Sunday, day_of_week_Thursday, day_of_week_Tuesday, day_of_week_Wednesday,
    # 22 station dummies, 4 city dummies
    input_list = [0.0] * 45
    
    # Set continuous and manual values
    input_list[0] = month
    input_list[1] = hour
    input_list[2] = is_weekend
    input_list[3] = o3
    input_list[4] = temp
    input_list[5] = humidity
    input_list[6] = wind_speed
    input_list[7] = visibility
    input_list[8] = gpi
    input_list[9] = pm_coarse
    input_list[10] = float(s_monsoon)  # Convert bool to float
    input_list[11] = float(s_summer)
    input_list[12] = float(s_winter)
    
    # Set day of week dummy (Friday is reference, all 0)
    day_mapping = {
        'day_of_week_Monday': 13,
        'day_of_week_Saturday': 14,
        'day_of_week_Sunday': 15,
        'day_of_week_Thursday': 16,
        'day_of_week_Tuesday': 17,
        'day_of_week_Wednesday': 18
    }
    if selected_day in day_mapping:
        input_list[day_mapping[selected_day]] = 1.0
    
    # Set station dummy
    station_start = 19
    for i, st_name in enumerate(stations):
        if st_name == selected_station:
            input_list[station_start + i] = 1.0
            break
    
    # Set city dummy
    city_start = 41
    for i, ct in enumerate(cities):
        if ct == selected_city:
            input_list[city_start + i] = 1.0
            break
    
    # Convert to numpy array
    final_input = np.array([input_list])
    
    try:
        # Scale -> Predict -> Inverse Scale
        scaled_input = scaler_x.transform(final_input)
        st.write(f"Debug - Scaled Input: {scaled_input}")  # Debug: Check scaled input
        
        prediction_scaled = model.predict(scaled_input)
        st.write(f"Debug - Model Prediction (Scaled): {prediction_scaled}")  # Debug: Check model output
        
        prediction_final = scaler_y.inverse_transform(prediction_scaled)
        st.write(f"Debug - Inverse Transformed Prediction: {prediction_final}")  # Debug: Check inverse transform
        
        aqi_res = prediction_final[0][0]
        # Clip AQI to valid range [0, 500]
        aqi_res = np.clip(aqi_res, 0, 500)
        
        st.success(f"### Predicted AQI: {aqi_res:.2f}")
        
        if aqi_res <= 100:
            st.balloons()
    except Exception as e:
        st.write(f"Prediction failed: {e}")
