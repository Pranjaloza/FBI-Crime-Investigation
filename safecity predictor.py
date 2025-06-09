import streamlit as st
import pandas as pd
import joblib
import datetime

# Load model
model = joblib.load(r"C:\Users\Pranjal Oza\crime_forecasting_xgb_model.pkl")

# Page Config
st.set_page_config(page_title="Crime Forecast Dashboard", page_icon="📊", layout="centered")

# Title
st.title("🚔SafeCity Predictor")

with st.container():
    st.markdown("#### 📅 Select Date & Crime Type")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input("Date", value=datetime.date(2025, 6, 9))

    with col2:
        crime_types = ['Theft from Vehicle', 'Mischief', 'Break and Enter Residential/Other',
                       'Theft of Vehicle', 'Other Theft', 'Vehicle Collision or Pedestrian Struck (with Injury)']
        crime_type = st.selectbox("Crime Type", crime_types)

    # Predict button
    if st.button("🚨 Predict Incident Count"):
        year, month = selected_date.year, selected_date.month

        # Prepare one-hot encoded input data
        input_data = {'YEAR': year, 'MONTH': month}
        for c in crime_types:
            input_data[f'TYPE_{c}'] = 1 if c == crime_type else 0

        # Fill missing columns
        model_features = model.get_booster().feature_names
        for col in model_features:
            if col not in input_data:
                input_data[col] = 0

        # Convert to DataFrame and reorder columns
        df = pd.DataFrame([input_data])
        df = df[model_features]

        # Make prediction
        prediction = model.predict(df)[0]

        # Show result
        st.success(f"📊 Predicted Incidents for {crime_type}: **{prediction:.0f}**")

# Footer line
st.markdown("---")
