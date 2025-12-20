        
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load artifacts
# model = joblib.load('best_model.pkl')
# scaler = joblib.load('scaler.pkl')
# features = joblib.load('feature_names.pkl')

# st.set_page_config(page_title="OkoaMaisha | LoS Predictor", layout="wide")

# st.title("🏥 OkoaMaisha: Clinical Length of Stay Predictor")
# st.markdown("Predict hospital stay duration to optimize bed resource management.")

# # Create two columns for input
# col1, col2 = st.columns(2)

# with col1:
#     st.header("Patient Demographics")
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     rcount = st.slider("Previous Readmissions", 0, 5, 0)
    
# with col2:
#     st.header("Clinical Biomarkers")
#     glucose = st.number_input("Glucose Level", value=100.0)
#     sodium = st.number_input("Sodium Level", value=135.0)
#     creatinine = st.number_input("Creatinine Level", value=1.0)

# # Preparation for prediction
# if st.button("Predict Length of Stay"):
#     # 1. Create a dictionary matching the training feature set
#     # Note: You must ensure all dummy columns (facid_B, etc.) are present
#     input_data = pd.DataFrame(0, index=[0], columns=features)
    
#     # 2. Map inputs to the dataframe
#     input_data['gender'] = 1 if gender == "Male" else 0
#     input_data['rcount'] = float(rcount)
#     # Check if these column names match your CSV exactly:
#     if 'glucose' in features: input_data['glucose'] = glucose
#     if 'sodium' in features: input_data['sodium'] = sodium
#     if 'creatinine' in features: input_data['creatinine'] = creatinine
    
#     # 3. Scale and Predict
#     scaled_input = scaler.transform(input_data)
#     prediction = model.predict(scaled_input)
    
#     # 4. Display Result
#     st.success(f"### Predicted Stay: {prediction[0]:.2f} Days")
    
#     # Resource Logic
#     if prediction[0] > 7:
#         st.warning("High Resource Intensity: Consider early discharge planning.")
#     else:
#         st.info("Standard Stay: Routine resource allocation.") 

"""
OkoaMaisha: Clinical Length of Stay Predictor
Streamlit Web Application - Complete Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================
# LOAD MODEL ARTIFACTS
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, scaler, feature_names, metadata

model, scaler, feature_names, metadata = load_model()

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="OkoaMaisha | LoS Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {font-size:20px !important; font-weight: bold;}
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.title("🏥 OkoaMaisha: Clinical Length of Stay Predictor")
st.markdown(f"""
**Predict hospital stay duration to optimize bed resource management**  
*Model: {metadata['model_name']} | Accuracy (R²): {metadata['test_r2']:.3f} | Avg Error: ±{metadata['test_mae']:.1f} days*
""")

st.markdown("---")

# ============================================
# SIDEBAR - MODEL INFO
# ============================================
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"""
    **Training Date:** {metadata['training_date']}  
    **Total Features:** {len(feature_names)}  
    **Performance:** R² = {metadata['test_r2']:.3f}
    
    This model predicts patient length of stay based on:
    - Demographics
    - Medical history (comorbidities)
    - Vital signs
    - Lab results
    - Admission details
    """)
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. Enter patient information
    2. Complete clinical measurements
    3. Click 'Predict Length of Stay'
    4. View resource recommendations
    """)

# ============================================
# INPUT FORM
# ============================================
st.header("📝 Patient Information")

# Create tabs for organized input
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Demographics", 
    "🩺 Medical History", 
    "💉 Vital Signs & Labs",
    "🏥 Admission Details"
])

# TAB 1: Demographics
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_encoded = 1 if gender == "Male" else 0
    
    with col2:
        rcount = st.slider("Previous Readmissions (rcount)", 0, 5, 0, 
                          help="Number of times patient was readmitted in past 180 days")
    
    with col3:
        bmi = st.number_input("BMI (Body Mass Index)", 
                             min_value=10.0, max_value=60.0, value=25.0, step=0.1)

# TAB 2: Medical History (Comorbidities)
with tab2:
    st.markdown("**Select all conditions that apply:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Cardiovascular & Renal**")
        dialysisrenalendstage = st.checkbox("Dialysis/End-stage Renal Disease")
        hemo = st.checkbox("Hemoglobin Disorder")
    
    with col2:
        st.markdown("**Respiratory & Metabolic**")
        asthma = st.checkbox("Asthma")
        pneum = st.checkbox("Pneumonia")
        irondef = st.checkbox("Iron Deficiency")
        malnutrition = st.checkbox("Malnutrition")
    
    with col3:
        st.markdown("**Mental Health & Substance**")
        psychologicaldisordermajor = st.checkbox("Major Psychological Disorder")
        depress = st.checkbox("Depression")
        psychother = st.checkbox("Other Psychological Condition")
        substancedependence = st.checkbox("Substance Dependence")
        fibrosisandother = st.checkbox("Fibrosis & Other Conditions")

# TAB 3: Vital Signs & Lab Results
with tab3:
    st.markdown("**Clinical Measurements**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Vital Signs**")
        pulse = st.number_input("Pulse (bpm)", 
                               min_value=40, max_value=180, value=75, step=1)
        respiration = st.number_input("Respiration Rate (breaths/min)", 
                                     min_value=8.0, max_value=40.0, value=16.0, step=0.5)
    
    with col2:
        st.markdown("**Blood Work**")
        hematocrit = st.number_input("Hematocrit (%)", 
                                    min_value=20.0, max_value=60.0, value=40.0, step=0.5)
        neutrophils = st.number_input("Neutrophils (cells/µL)", 
                                     min_value=0.0, max_value=20.0, value=4.0, step=0.1)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 
                                 min_value=50.0, max_value=400.0, value=100.0, step=1.0)
        sodium = st.number_input("Sodium (mEq/L)", 
                                min_value=120.0, max_value=160.0, value=140.0, step=0.5)
    
    with col2:
        creatinine = st.number_input("Creatinine (mg/dL)", 
                                    min_value=0.3, max_value=10.0, value=1.0, step=0.1)
        bloodureanitro = st.number_input("Blood Urea Nitrogen (mg/dL)", 
                                        min_value=5.0, max_value=100.0, value=12.0, step=0.5)

# TAB 4: Admission Details
with tab4:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        facility = st.selectbox("Hospital Facility", ["A", "B", "C", "D", "E"])
    
    with col2:
        admission_month = st.selectbox("Admission Month", 
                                      list(range(1, 13)), 
                                      format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
    
    with col3:
        admission_dayofweek = st.selectbox("Day of Week", 
                                          ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                           "Friday", "Saturday", "Sunday"])
        dayofweek_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                        "Friday": 4, "Saturday": 5, "Sunday": 6}
        admission_dayofweek_encoded = dayofweek_map[admission_dayofweek]
    
    admission_quarter = (admission_month - 1) // 3 + 1
    
    secondarydiagnosisnonicd9 = st.slider("Secondary Diagnoses Count", 
                                         0, 10, 1, 
                                         help="Number of secondary diagnoses (complexity indicator)")

# ============================================
# PREDICTION
# ============================================
st.markdown("---")
st.header("🔮 Prediction")

if st.button("🚀 Predict Length of Stay", type="primary", use_container_width=True):
    # Create input dataframe matching training features
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in the values
    input_data['gender'] = gender_encoded
    input_data['rcount'] = rcount
    input_data['dialysisrenalendstage'] = int(dialysisrenalendstage)
    input_data['asthma'] = int(asthma)
    input_data['irondef'] = int(irondef)
    input_data['pneum'] = int(pneum)
    input_data['substancedependence'] = int(substancedependence)
    input_data['psychologicaldisordermajor'] = int(psychologicaldisordermajor)
    input_data['depress'] = int(depress)
    input_data['psychother'] = int(psychother)
    input_data['fibrosisandother'] = int(fibrosisandother)
    input_data['malnutrition'] = int(malnutrition)
    input_data['hemo'] = int(hemo)
    input_data['hematocrit'] = hematocrit
    input_data['neutrophils'] = neutrophils
    input_data['sodium'] = sodium
    input_data['glucose'] = glucose
    input_data['bloodureanitro'] = bloodureanitro
    input_data['creatinine'] = creatinine
    input_data['bmi'] = bmi
    input_data['pulse'] = pulse
    input_data['respiration'] = respiration
    input_data['secondarydiagnosisnonicd9'] = secondarydiagnosisnonicd9
    input_data['admission_month'] = admission_month
    input_data['admission_dayofweek'] = admission_dayofweek_encoded
    input_data['admission_quarter'] = admission_quarter
    
    # Engineered features
    comorbidity_count = sum([dialysisrenalendstage, asthma, irondef, pneum, 
                             substancedependence, psychologicaldisordermajor,
                             depress, psychother, fibrosisandother, malnutrition, hemo])
    input_data['total_comorbidities'] = comorbidity_count
    input_data['high_glucose'] = int(glucose > 140)
    input_data['low_sodium'] = int(sodium < 135)
    input_data['high_creatinine'] = int(creatinine > 1.3)
    input_data['low_bmi'] = int(bmi < 18.5)
    input_data['high_bmi'] = int(bmi > 30)
    input_data['abnormal_vitals'] = int((pulse < 60 or pulse > 100)) + int((respiration < 12 or respiration > 20))
    
    # Facility encoding
    for fac in ['A', 'B', 'C', 'D', 'E']:
        col_name = f'facility_{fac}'
        if col_name in feature_names:
            input_data[col_name] = int(facility == fac)
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # Display results
    st.success("### ✅ Prediction Complete")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Length of Stay", f"{prediction:.1f} days", 
                 delta=f"±{metadata['test_mae']:.1f} days (model error)")
    
    with col2:
        if prediction <= 3:
            category = "Short Stay"
            color = "🟢"
        elif prediction <= 7:
            category = "Medium Stay"
            color = "🟡"
        else:
            category = "Long Stay"
            color = "🔴"
        st.metric("Stay Category", f"{color} {category}")
    
    with col3:
        st.metric("Comorbidity Burden", f"{comorbidity_count} conditions")
    
    # Resource recommendations
    st.markdown("---")
    st.subheader("📋 Resource Planning Recommendations")
    
    if prediction > 7:
        st.warning(f"""
        **High Resource Intensity** (Predicted {prediction:.1f} days)
        
        ✓ **Bed Management:** Reserve extended care bed  
        ✓ **Staff Planning:** Allocate specialized nursing staff  
        ✓ **Supply Chain:** Ensure 10-day medication supply  
        ✓ **Discharge Planning:** Initiate early planning for post-acute care  
        ✓ **Care Coordination:** Schedule case management review
        """)
    elif prediction > 4:
        st.info(f"""
        **Standard Resource Allocation** (Predicted {prediction:.1f} days)
        
        ✓ **Bed Management:** Standard acute care bed  
        ✓ **Staff Planning:** Regular staffing levels  
        ✓ **Supply Chain:** Standard 7-day supply protocol  
        ✓ **Monitoring:** Routine progress assessments
        """)
    else:
        st.success(f"""
        **Low Resource Intensity** (Predicted {prediction:.1f} days)
        
        ✓ **Bed Management:** Short-stay or observation unit eligible  
        ✓ **Staff Planning:** Standard staffing sufficient  
        ✓ **Supply Chain:** Minimal supply requirements  
        ✓ **Discharge:** Early discharge planning opportunity
        """)
    
    # Risk factors
    if comorbidity_count > 2:
        st.warning(f"⚠️ **High Comorbidity Burden:** Patient has {comorbidity_count} concurrent conditions")
    
    if glucose > 140:
        st.warning("⚠️ **Elevated Glucose:** Consider diabetes management protocol")
    
    if sodium < 135:
        st.warning("⚠️ **Hyponatremia:** Monitor electrolyte balance closely")
    
    if creatinine > 1.3:
        st.warning("⚠️ **Elevated Creatinine:** Kidney function monitoring required")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>OkoaMaisha</strong> - Hospital Resource Optimization System</p>
    <p style='font-size: 12px;'>This tool is for clinical decision support only. 
    Final decisions should be made by qualified healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
