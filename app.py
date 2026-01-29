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

# --- 2. Feature Definitions from your Image ---
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
    selected_day = st.selectbox("Day of Week", ["Friday"] + days_of_week) # Friday is likely the dropped column
    
    st.header("🗓️ Season")
    # Using bool logic for seasons
    s_summer = st.checkbox("Summer")
    s_winter = st.checkbox("Winter")
    s_monsoon = st.checkbox("Post Monsoon")

st.subheader("🧪 Pollutant Concentrations & Weather")
col1, col2, col3 = st.columns(3)

with col1:
    month = st.slider("Month", 1, 12, 1)
    hour = st.slider("Hour", 0, 23, 12)
    is_weekend = st.radio("Is Weekend?", [0.0, 1.0])

with col2:
    o3 = st.number_input("O3", value=0.0)
    temp = st.number_input("Temperature", value=25.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    wind = st.number_input("Wind Speed", value=5.0)

with col3:
    vis = st.number_input("Visibility", value=10.0)
    gpi = st.number_input("GPI (Gaseous Pollutant Index)", value=0.0)
    pm_coarse = st.number_input("PM Coarse", value=0.0)

# --- 4. Prediction Logic ---
if st.button("Predict Air Quality Index"):
    # 1. Initialize a dictionary with all 47 features set to 0/False
    # Based on your image: 47 total columns
    input_dict = {col: 0.0 for col in range(47)} 
    
    # We must map input names to their EXACT column index from your image
    # Mapping based on image_1969f6.png column indices:
    features = {
        0: month, 1: hour, 2: is_weekend, 3: o3, 4: temp, 5: humidity,
        6: wind_speed, 7: visibility, 10: gpi, 11: pm_coarse,
        12: s_monsoon, 13: s_summer, 14: s_winter
    }
    
    # 2. Update Continuous/Manual values
    for idx, val in features.items():
        input_dict[idx] = val
        
    # 3. Handle Dropdowns (Setting selected to 1.0)
    # We find the index of the selected string in the full list and set it
    for i, col_name in enumerate(days_of_week, start=15): # Days start at index 15
        if col_name == selected_day: input_dict[i] = 1.0
        
    for i, col_name in enumerate(stations, start=21): # Stations start at index 21
        if col_name == selected_station: input_dict[i] = 1.0
        
    for i, col_name in enumerate(cities, start=43): # Cities start at index 43
        if col_name == selected_city: input_dict[i] = 1.0

    # Convert dict to array and ensure correct shape (1, 47)
    final_input = np.array([list(input_dict.values())])
    
    try:
        # Scale -> Predict -> Inverse Scale
        scaled_input = scaler_x.transform(final_input)
        prediction_scaled = model.predict(scaled_input)
        prediction_final = scaler_y.inverse_transform(prediction_scaled)
        
        aqi_res = prediction_final[0][0]
        st.success(f"### Predicted AQI: {aqi_res:.2f}")
        
        if aqi_res <= 100: st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
