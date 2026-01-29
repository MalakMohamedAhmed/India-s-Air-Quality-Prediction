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
# Based on DataFrame structure from screenshots:
# Indices 0-7: month, hour, is_weekend, o3, temperature, humidity, wind_speed, visibility
# Index 8: aqi (target - NOT a feature)
# Index 9: aqi_category (target - NOT a feature)
# Indices 10-50: GPI, pm_coarse, season features, day of week features, station features, city features
# Total features for input: 49 (0-7, 10-50)

# Lists for dummy variables to be handled by dropdowns
stations = [
    'station_Anand Vihar, Delhi', 'station_Bawana, Delhi', 'station_Dwarka Sec 8, Delhi',
    'station_Faridabad New Town', 'station_Faridabad Sec 16A', 'station_Ghaziabad Loni',
    'station_Ghaziabad Vasundhara', 'station_Greater Noida', 'station_Gurugram Sec 51',
    'station_Gurugram Vikas Sadan', 'station_ITO, Delhi', 'station_Jahangirpuri, Delhi',
    'station_Mandir Marg, Delhi', 'station_NSIT Dwarka, Delhi', 'station_Noida Sec 125',
    'station_Noida Sec 62', 'station_Okhla Phase 2, Delhi', 'station_Punjabi Bagh, Delhi',
    'station_RK Puram, Delhi', 'station_Rohini, Delhi', 'station_Shadipur, Delhi',
    'station_Siri Fort, Delhi', 'station_Wazirpur, Delhi'
]

cities = ['city_Delhi', 'city_Faridabad', 'city_Ghaziabad', 'city_Gurugram', 'city_Noida']

days_of_week = [
    'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday',
    'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
]

seasons = ['season_monsoon', 'season_post_monsoon', 'season_summer', 'season_winter']

# --- 3. UI Layout ---
st.set_page_config(page_title="India AQI Predictor", layout="wide")
st.title("🌍 Indian Air Quality Prediction (ANN)")

with st.sidebar:
    st.header("📍 Location & Time")
    selected_city = st.selectbox("Select City", cities)
    selected_station = st.selectbox("Select Station", stations)
    selected_day = st.selectbox("Day of Week", ["Friday"] + days_of_week)  # Friday is the reference (all dummies 0)
    
    st.header("🗓️ Season")
    # Using radio button for season selection (only one can be active)
    season_options = ["None"] + seasons
    selected_season = st.radio("Select Season", season_options)

st.subheader("🧪 Pollutant Concentrations & Weather")
col1, col2, col3 = st.columns(3)

with col1:
    month = st.slider("Month", 1, 12, 1)
    hour = st.slider("Hour", 0, 23, 12)
    is_weekend = st.radio("Is Weekend?", [0.0, 1.0])

with col2:
    o3 = st.number_input("O3", value=0.0, min_value=0.0)
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
    
    # Initialize input list with 49 features (excluding aqi at index 8 and aqi_category at index 9)
    # Structure based on DataFrame:
    # 0: month, 1: hour, 2: is_weekend, 3: o3, 4: temperature, 5: humidity, 6: wind_speed, 7: visibility
    # [skip 8: aqi, 9: aqi_category]
    # 10: GPI, 11: pm_coarse
    # 12-15: season_monsoon, season_post_monsoon, season_summer, season_winter
    # 16-21: day_of_week_Monday, day_of_week_Saturday, day_of_week_Sunday, day_of_week_Thursday, day_of_week_Tuesday, day_of_week_Wednesday
    # 22-44: 23 station dummies
    # 45-49: 5 city dummies
    
    input_list = [0.0] * 49
    
    # Set continuous values (indices 0-7)
    input_list[0] = float(month)
    input_list[1] = float(hour)
    input_list[2] = float(is_weekend)
    input_list[3] = float(o3)
    input_list[4] = float(temp)
    input_list[5] = float(humidity)
    input_list[6] = float(wind_speed)
    input_list[7] = float(visibility)
    
    # Set GPI and pm_coarse (indices 8-9 in our feature array, which maps to 10-11 in DataFrame)
    input_list[8] = float(gpi)
    input_list[9] = float(pm_coarse)
    
    # Set season dummies (indices 10-13 in our feature array, which maps to 12-15 in DataFrame)
    season_mapping = {
        'season_monsoon': 10,
        'season_post_monsoon': 11,
        'season_summer': 12,
        'season_winter': 13
    }
    if selected_season in season_mapping:
        input_list[season_mapping[selected_season]] = 1.0
    
    # Set day of week dummy (indices 14-19 in our feature array, which maps to 16-21 in DataFrame)
    # Friday is reference, all 0
    day_mapping = {
        'day_of_week_Monday': 14,
        'day_of_week_Saturday': 15,
        'day_of_week_Sunday': 16,
        'day_of_week_Thursday': 17,
        'day_of_week_Tuesday': 18,
        'day_of_week_Wednesday': 19
    }
    if selected_day in day_mapping:
        input_list[day_mapping[selected_day]] = 1.0
    
    # Set station dummy (indices 20-42 in our feature array, which maps to 22-44 in DataFrame)
    station_start = 20
    for i, st_name in enumerate(stations):
        if st_name == selected_station:
            input_list[station_start + i] = 1.0
            break
    
    # Set city dummy (indices 43-47 in our feature array, which maps to 45-49 in DataFrame)
    city_start = 43
    for i, ct in enumerate(cities):
        if ct == selected_city:
            input_list[city_start + i] = 1.0
            break
    
    # Convert to numpy array
    final_input = np.array([input_list])
    
    # Display debug info
    with st.expander("🔍 Debug Information"):
        st.write(f"**Input Shape:** {final_input.shape}")
        st.write(f"**Expected Features:** 49")
        st.write(f"**Actual Features:** {len(input_list)}")
        st.write(f"**Non-zero Features:**")
        non_zero_indices = np.where(final_input[0] != 0)[0]
        for idx in non_zero_indices:
            st.write(f"  - Index {idx}: {final_input[0][idx]}")
    
    try:
        # Scale -> Predict -> Inverse Scale
        scaled_input = scaler_x.transform(final_input)
        
        with st.expander("🔬 Prediction Pipeline"):
            st.write(f"**Scaled Input Shape:** {scaled_input.shape}")
            st.write(f"**Scaled Input (first 10 values):** {scaled_input[0][:10]}")
        
        prediction_scaled = model.predict(scaled_input, verbose=0)
        
        with st.expander("🔬 Prediction Pipeline"):
            st.write(f"**Model Prediction (Scaled):** {prediction_scaled[0][0]:.4f}")
        
        prediction_final = scaler_y.inverse_transform(prediction_scaled)
        
        with st.expander("🔬 Prediction Pipeline"):
            st.write(f"**Inverse Transformed Prediction:** {prediction_final[0][0]:.4f}")
        
        aqi_res = prediction_final[0][0]
        # Clip AQI to valid range [0, 500]
        aqi_res = np.clip(aqi_res, 0, 500)
        
        # Display result with color coding
        st.markdown("---")
        if aqi_res <= 50:
            st.success(f"### 🟢 Predicted AQI: {aqi_res:.2f} - Good")
            st.balloons()
        elif aqi_res <= 100:
            st.success(f"### 🟡 Predicted AQI: {aqi_res:.2f} - Satisfactory")
        elif aqi_res <= 200:
            st.warning(f"### 🟠 Predicted AQI: {aqi_res:.2f} - Moderate")
        elif aqi_res <= 300:
            st.warning(f"### 🔴 Predicted AQI: {aqi_res:.2f} - Poor")
        elif aqi_res <= 400:
            st.error(f"### 🟣 Predicted AQI: {aqi_res:.2f} - Very Poor")
        else:
            st.error(f"### ⚫ Predicted AQI: {aqi_res:.2f} - Severe")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.write("**Error Details:**")
        st.write(f"Input shape: {final_input.shape}")
        st.write(f"Expected by scaler: {scaler_x.n_features_in_}")
