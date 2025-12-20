import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="OkoaMaisha | LoS Predictor", layout="wide")

st.title("🏥 OkoaMaisha: Clinical Length of Stay Predictor")
st.markdown("Predict hospital stay duration to optimize bed resource management.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    rcount = st.slider("Previous Readmissions", 0, 5, 0)
    
with col2:
    st.header("Clinical Biomarkers")
    glucose = st.number_input("Glucose Level", value=100.0)
    sodium = st.number_input("Sodium Level", value=135.0)
    creatinine = st.number_input("Creatinine Level", value=1.0)

# Preparation for prediction
if st.button("Predict Length of Stay"):
    # 1. Create a dictionary matching the training feature set
    # Note: You must ensure all dummy columns (facid_B, etc.) are present
    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    # 2. Map inputs to the dataframe
    input_data['gender'] = 1 if gender == "Male" else 0
    input_data['rcount'] = float(rcount)
    # Check if these column names match your CSV exactly:
    if 'glucose' in features: input_data['glucose'] = glucose
    if 'sodium' in features: input_data['sodium'] = sodium
    if 'creatinine' in features: input_data['creatinine'] = creatinine
    
    # 3. Scale and Predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    # 4. Display Result
    st.success(f"### Predicted Stay: {prediction[0]:.2f} Days")
    
    # Resource Logic
    if prediction[0] > 7:
        st.warning("High Resource Intensity: Consider early discharge planning.")
    else:
        st.info("Standard Stay: Routine resource allocation.")
