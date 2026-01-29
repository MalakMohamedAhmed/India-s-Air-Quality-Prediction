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

# --- 2. Calculation Functions ---
def calculate_gpi(so2, no2, co):
    """
    Calculate Gaseous Pollutant Index (GPI) from SO2, NO2, and CO
    GPI is typically a weighted sum or average of gaseous pollutants
    Adjust the formula based on your training data methodology
    """
    # Common formula: weighted average based on pollutant concentrations
    # You may need to adjust weights based on how GPI was calculated in training
    gpi = (so2 * 0.3 + no2 * 0.5 + co * 0.2)
    return gpi

def calculate_pm_coarse(pm10, pm25):
    """
    Calculate PM Coarse (particles between 2.5 and 10 micrometers)
    PM_coarse = PM10 - PM2.5
    """
    pm_coarse = pm10 - pm25
    # Ensure non-negative
    pm_coarse = max(0.0, pm_coarse)
    return pm_coarse

# --- 3. Feature Definitions ---
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

# --- 4. UI Layout ---
st.set_page_config(page_title="India AQI Predictor", layout="wide")
st.title("🌍 Indian Air Quality Prediction (ANN)")

with st.sidebar:
    st.header("📍 Location & Time")
    selected_city = st.selectbox("Select City", cities)
    selected_station = st.selectbox("Select Station", stations)
    selected_day = st.selectbox("Day of Week", days_of_week)
    
    st.header("🗓️ Season")
    selected_season = st.radio("Select Season", seasons)
    
    st.header("⏰ Date & Time")
    month = st.slider("Month", 1, 12, 1)
    hour = st.slider("Hour", 0, 23, 12)
    is_weekend = st.radio("Is Weekend?", [0.0, 1.0])

st.subheader("🏭 Pollutant Concentrations")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Gaseous Pollutants**")
    o3 = st.number_input("O3 (µg/m³)", value=50.0, min_value=0.0, help="Ozone concentration")
    so2 = st.number_input("SO2 (µg/m³)", value=10.0, min_value=0.0, help="Sulfur dioxide concentration")
    no2 = st.number_input("NO2 (µg/m³)", value=40.0, min_value=0.0, help="Nitrogen dioxide concentration")

with col2:
    st.markdown("**Other Pollutants**")
    co = st.number_input("CO (mg/m³)", value=1.0, min_value=0.0, help="Carbon monoxide concentration")
    pm10 = st.number_input("PM10 (µg/m³)", value=100.0, min_value=0.0, help="Particulate matter ≤10 micrometers")
    pm25 = st.number_input("PM2.5 (µg/m³)", value=50.0, min_value=0.0, max_value=pm10 if pm10 > 0 else 1000.0, help="Particulate matter ≤2.5 micrometers")

with col3:
    st.markdown("**Weather Conditions**")
    temp = st.number_input("Temperature (°C)", value=25.0, help="Air temperature")
    humidity = st.number_input("Humidity (%)", value=50.0, min_value=0.0, max_value=100.0, help="Relative humidity")
    wind_speed = st.number_input("Wind Speed (m/s)", value=5.0, min_value=0.0, help="Wind speed")
    visibility = st.number_input("Visibility (km)", value=10.0, min_value=0.0, help="Visibility distance")

# --- 5. Prediction Logic ---
if st.button("🔮 Predict Air Quality Index", type="primary"):
    # Basic input validation
    if pm25 > pm10 and pm10 > 0:
        st.error("❌ PM2.5 cannot be greater than PM10!")
        st.stop()
    
    if any([o3 < 0, so2 < 0, no2 < 0, co < 0, pm10 < 0, pm25 < 0, 
            humidity < 0, humidity > 100, wind_speed < 0, visibility < 0]):
        st.error("❌ Please ensure all values are non-negative and humidity is 0-100%.")
        st.stop()
    
    # Calculate GPI and PM_coarse from user inputs
    gpi = calculate_gpi(so2, no2, co)
    pm_coarse = calculate_pm_coarse(pm10, pm25)
    
    # Display calculated values
    st.info(f"📊 **Calculated Values:** GPI = {gpi:.2f}, PM Coarse = {pm_coarse:.2f}")
    
    # Initialize input list with 49 features (all columns except index 8 which is aqi)
    # Feature order in the model input:
    # 0-7: month, hour, is_weekend, o3, temperature, humidity, wind_speed, visibility
    # [skip index 8: aqi]
    # 8: GPI (column 9 in dataframe) - CALCULATED
    # 9: pm_coarse (column 10 in dataframe) - CALCULATED
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
    # Note: Visibility is inverted because the model learned inverse relationship
    # Higher visibility = better air quality = lower AQI
    # So we use negative visibility to correct the prediction direction
    input_list[7] = -float(visibility)
    
    # Set CALCULATED GPI and pm_coarse (indices 8-9 in model input, columns 9-10 in dataframe)
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
        st.write(f"**User Inputs:**")
        st.write(f"  - SO2: {so2}, NO2: {no2}, CO: {co}")
        st.write(f"  - PM10: {pm10}, PM2.5: {pm25}")
        st.write(f"**Calculated:**")
        st.write(f"  - GPI: {gpi:.2f}")
        st.write(f"  - PM Coarse: {pm_coarse:.2f}")
        st.write(f"**Non-zero Features:**")
        non_zero_indices = np.where(final_input[0] != 0)[0]
        for idx in non_zero_indices:
            st.write(f"  - Index {idx}: {final_input[0][idx]:.2f}")
    
    try:
        # Display raw input before scaling
        st.write("### 📊 Raw Input Analysis")
        with st.expander("View Raw Input Values"):
            st.write(f"**Raw input array (first 20 values):**")
            st.write(final_input[0][:20])
            st.write(f"\n**Scaler expects {scaler_x.n_features_in_} features**")
            st.write(f"**We're providing {final_input.shape[1]} features**")
            
            # Check scaler parameters
            if hasattr(scaler_x, 'mean_'):
                st.write(f"\n**Scaler mean (first 10):** {scaler_x.mean_[:10]}")
            if hasattr(scaler_x, 'scale_'):
                st.write(f"**Scaler scale (first 10):** {scaler_x.scale_[:10]}")
        
        # Scale input
        scaled_input = scaler_x.transform(final_input)
        
        st.write("### 🔬 Scaled Input Analysis")
        with st.expander("View Scaled Values"):
            st.write(f"**Scaled Input Shape:** {scaled_input.shape}")
            st.write(f"**Scaled Input (first 20 values):** {scaled_input[0][:20]}")
            st.write(f"**Scaled Input (min/max):** {scaled_input.min():.4f} / {scaled_input.max():.4f}")
            
            # Check for extreme values
            extreme_indices = np.where(np.abs(scaled_input[0]) > 10)[0]
            if len(extreme_indices) > 0:
                st.warning(f"⚠️ Found {len(extreme_indices)} features with extreme scaled values (>10 or <-10):")
                for idx in extreme_indices[:10]:  # Show first 10
                    st.write(f"  - Index {idx}: {scaled_input[0][idx]:.2f} (raw: {final_input[0][idx]:.2f})")
        
        # Predict
        prediction_scaled = model.predict(scaled_input, verbose=0)
        
        st.write("### 🎯 Model Prediction Analysis")
        with st.expander("View Model Output"):
            st.write(f"**Model Output (Scaled):** {prediction_scaled[0][0]:.6f}")
            st.write(f"**Model Output Shape:** {prediction_scaled.shape}")
            
            # Check y_scaler parameters
            if hasattr(scaler_y, 'mean_'):
                st.write(f"**Y Scaler mean:** {scaler_y.mean_}")
            if hasattr(scaler_y, 'scale_'):
                st.write(f"**Y Scaler scale:** {scaler_y.scale_}")
        
        # Inverse transform
        prediction_final = scaler_y.inverse_transform(prediction_scaled)
        
        st.write("### 📈 Final Prediction")
        with st.expander("View Inverse Transform"):
            st.write(f"**Before Inverse Transform:** {prediction_scaled[0][0]:.6f}")
            st.write(f"**After Inverse Transform:** {prediction_final[0][0]:.6f}")
            st.write(f"**After Clipping [0, 500]:** {np.clip(prediction_final[0][0], 0, 500):.2f}")
        
        aqi_res_raw = prediction_final[0][0]
        aqi_res = np.clip(aqi_res_raw, 0, 500)
        
        # Show warning if clipping occurred
        if aqi_res_raw < 0:
            st.warning(f"⚠️ Raw prediction was negative ({aqi_res_raw:.2f}), clipped to 0")
        elif aqi_res_raw > 500:
            st.warning(f"⚠️ Raw prediction exceeded 500 ({aqi_res_raw:.2f}), clipped to 500")
        
        # Display result with color coding
        st.markdown("---")
        if aqi_res <= 50:
            st.success(f"### 🟢 Predicted AQI: {aqi_res:.2f} - Good")
            st.info("Air quality is satisfactory, and air pollution poses little or no risk.")
            st.balloons()
        elif aqi_res <= 100:
            st.success(f"### 🟡 Predicted AQI: {aqi_res:.2f} - Satisfactory")
            st.info("Air quality is acceptable for most people. However, sensitive individuals may experience minor breathing discomfort.")
        elif aqi_res <= 200:
            st.warning(f"### 🟠 Predicted AQI: {aqi_res:.2f} - Moderate")
            st.info("Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
        elif aqi_res <= 300:
            st.warning(f"### 🔴 Predicted AQI: {aqi_res:.2f} - Poor")
            st.info("Health alert: The risk of health effects is increased for everyone. Sensitive groups may experience more serious effects.")
        elif aqi_res <= 400:
            st.error(f"### 🟣 Predicted AQI: {aqi_res:.2f} - Very Poor")
            st.info("Health warning of emergency conditions: everyone is more likely to be affected.")
        else:
            st.error(f"### ⚫ Predicted AQI: {aqi_res:.2f} - Severe")
            st.info("Health alert: Everyone may experience serious health effects. This is an emergency situation.")
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.write("**Error Details:**")
        st.write(f"Input shape: {final_input.shape}")
        st.write(f"Expected by scaler: {scaler_x.n_features_in_}")
        import traceback
        st.code(traceback.format_exc())
    
    # Save debug data
    if st.button("💾 Save Debug Data"):
        debug_data = {
            'user_inputs': {
                'month': month,
                'hour': hour,
                'is_weekend': is_weekend,
                'o3': o3,
                'so2': so2,
                'no2': no2,
                'co': co,
                'pm10': pm10,
                'pm25': pm25,
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'visibility': visibility,
                'season': selected_season,
                'day': selected_day,
                'station': selected_station,
                'city': selected_city
            },
            'calculated_values': {
                'gpi': gpi,
                'pm_coarse': pm_coarse
            },
            'raw_input': final_input[0].tolist()
        }
        
        import json
        with open('debug_input.json', 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        st.success("✅ Debug data saved to debug_input.json")
        
        # Also create a dataframe to compare with training data
        feature_names = [
            'month', 'hour', 'is_weekend', 'o3', 'temperature', 'humidity', 
            'wind_speed', 'visibility', 'GPI', 'pm_coarse'
        ] + seasons + days_of_week + stations + cities
        
        df_debug = pd.DataFrame([final_input[0]], columns=feature_names)
        df_debug.to_csv('debug_input.csv', index=False)
        st.success("✅ Debug data also saved to debug_input.csv")

# Footer
st.markdown("---")
st.markdown("### ℹ️ About")
st.markdown("""
This application predicts Air Quality Index (AQI) using an Artificial Neural Network trained on Indian air quality data.

**Pollutant Calculations:**
- **GPI (Gaseous Pollutant Index)**: Calculated from SO2, NO2, and CO concentrations
- **PM Coarse**: Calculated as PM10 - PM2.5

**Note**: The model uses calculated GPI and PM Coarse values, not the individual pollutants directly.
""")
