import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

@st.cache_resource
def load_assets():
    # Use compile=False to avoid the 'mse' deserialization error
    model = tf.models.load_model("model.h5", compile=False)
    scaler_x = joblib.load("scaler_x.sav")
    scaler_y = joblib.load("scaler_y.sav")
    return model, scaler_x, scaler_y

model, scaler_x, scaler_y = load_assets()

# --- 2. Feature Definitions ---
# Indices 0-7: month, hour, is_weekend, o3, temperature, humidity, wind_speed, visibility
# Index 8: aqi (TARGET - NOT a feature)
# Indices 9-49: GPI, pm_coarse, season features (4), day of week features (7), station features (23), city features (5)
# Total features for input: 49 (all columns except index 8)

# Lists for dummy variables
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
    'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
    'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday',
    'day_of_week_Wednesday'
]

seasons = ['season_monsoon', 'season_post_monsoon', 'season_summer', 'season_winter']

# --- 3. UI Layout ---
st.set_page_config(page_title="India AQI Predictor", layout="wide")
st.title("🌍 Indian Air Quality Prediction (ANN)")

with st.sidebar:
    st.header("📍 Location & Time")
    selected_city = st.selectbox("Select City", cities)
    selected_station = st.selectbox("Select Station", stations)
    selected_day = st.selectbox("Day of Week", days_of_week)
    
    st.header("🗓️ Season")
    selected_season = st.radio("Select Season", seasons)

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
    
    # Initialize input list with 49 features (all columns except index 8 which is aqi)
    # Feature order in the model input:
    # 0-7: month, hour, is_weekend, o3, temperature, humidity, wind_speed, visibility
    # [skip index 8: aqi]
    # 8: GPI (column 9 in dataframe)
    # 9: pm_coarse (column 10 in dataframe)
    # 10-13: season_monsoon, season_post_monsoon, season_summer, season_winter (columns 11-14)
    # 14-20: day_of_week_Friday through Wednesday (columns 15-21)
    # 21-43: 23 station dummies (columns 22-44)
    # 44-48: 5 city dummies (columns 45-49)
    
    input_list = [0.0] * 49
    
    # Set continuous values (indices 0-7 - same in both dataframe and model input)
    input_list[0] = float(month)
    input_list[1] = float(hour)
    input_list[2] = float(is_weekend)
    input_list[3] = float(o3)
    input_list[4] = float(temp)
    input_list[5] = float(humidity)
    input_list[6] = float(wind_speed)
    input_list[7] = float(visibility)
    
    # Set GPI and pm_coarse (indices 8-9 in model input, columns 9-10 in dataframe)
    input_list[8] = float(gpi)
    input_list[9] = float(pm_coarse)
    
    # Set season dummies (indices 10-13 in model input, columns 11-14 in dataframe)
    season_mapping = {
        'season_monsoon': 10,
        'season_post_monsoon': 11,
        'season_summer': 12,
        'season_winter': 13
    }
    if selected_season in season_mapping:
        input_list[season_mapping[selected_season]] = 1.0
    
    # Set day of week dummy (indices 14-20 in model input, columns 15-21 in dataframe)
    day_mapping = {
        'day_of_week_Friday': 14,
        'day_of_week_Monday': 15,
        'day_of_week_Saturday': 16,
        'day_of_week_Sunday': 17,
        'day_of_week_Thursday': 18,
        'day_of_week_Tuesday': 19,
        'day_of_week_Wednesday': 20
    }
    if selected_day in day_mapping:
        input_list[day_mapping[selected_day]] = 1.0
    
    # Set station dummy (indices 21-43 in model input, columns 22-44 in dataframe)
    station_start = 21
    for i, st_name in enumerate(stations):
        if st_name == selected_station:
            input_list[station_start + i] = 1.0
            break
    
    # Set city dummy (indices 44-48 in model input, columns 45-49 in dataframe)
    city_start = 44
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
            st.write(f"  - Index {idx}: {final_input[0][idx]:.2f}")
    
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
        import traceback
        st.code(traceback.format_exc())
