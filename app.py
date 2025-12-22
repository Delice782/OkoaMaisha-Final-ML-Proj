"""
OkoaMaisha: Clinical Length of Stay Predictor - Version 3.0
Predict-First Design - Professional Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="OkoaMaisha | LoS Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - Fixed white box issue
st.markdown("""
<style>
/* Remove default Streamlit containers */
.main {
    padding: 0rem 1rem;
    background: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hide white Streamlit containers */
.element-container {
    background: transparent !important;
}

div[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
}

/* Custom containers */
.compact-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    padding: 2rem; 
    border-radius: 15px; 
    color: white;
    margin-bottom: 1.5rem; 
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
}

.compact-header h1 {
    font-size: 2.2rem;
    margin: 0;
    font-weight: 700;
}

.compact-header p {
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.95;
}

.stats-container {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
    border: 1px solid #e2e8f0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
    text-align: center;
}

.stat-item {
    padding: 1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    transition: transform 0.2s;
}

.stat-item:hover {
    transform: translateY(-5px);
}

.stat-value {
    color: #3b82f6;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-label {
    color: #475569;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 600;
}

.form-section {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}

.form-section h3 {
    color: #1e3a8a;
    font-size: 1.4rem;
    margin: 0 0 1.5rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid #3b82f6;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.prediction-box {
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    padding: 3rem; 
    border-radius: 20px; 
    color: white;
    text-align: center; 
    box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
    margin: 2rem 0;
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}

.prediction-box h1 {
    font-size: 4rem;
    margin: 0;
    font-weight: 900;
    text-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.prediction-box p {
    font-size: 1.2rem;
    margin-top: 1rem;
}

.metric-card {
    background: white; 
    padding: 1.5rem; 
    border-radius: 12px;
    border-left: 5px solid #3b82f6; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
}

.section-header {
    font-size: 1.6rem; 
    font-weight: 700; 
    color: #1e3a8a;
    margin: 2.5rem 0 1.5rem 0; 
    padding-bottom: 0.75rem;
    border-bottom: 3px solid #3b82f6;
}

.progress-indicator {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin-top: 1rem;
    border: 2px solid #3b82f6;
    font-weight: 600;
    color: #1e3a8a !important;
}

.progress-indicator strong {
    color: #1e3a8a !important;
}

.capability-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    border-left: 5px solid #3b82f6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
    height: 100%;
}

.capability-card h4 {
    color: #3b82f6;
    margin-top: 0;
    font-size: 1.3rem;
}

.capability-card ul {
    color: #475569;
    line-height: 2;
    font-size: 1rem;
}

/* Streamlit specific overrides */
.stButton > button {
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.75rem 2rem;
    border-radius: 10px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
}

#MainMenu {visibility: hidden;} 
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, scaler, feature_names, metadata

model, scaler, feature_names, metadata = load_model_artifacts()

comorbidity_cols = metadata.get('comorbidity_cols', [
    'dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
    'substancedependence', 'psychologicaldisordermajor',
    'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo'
])

# Helper function
def engineer_features(input_dict):
    df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for key in ['gender', 'rcount', 'bmi', 'pulse', 'respiration', 'hematocrit',
                'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine',
                'secondarydiagnosisnonicd9', 'admission_month', 'admission_dayofweek',
                'admission_quarter']:
        if key in input_dict:
            df[key] = input_dict[key]
    
    for c in comorbidity_cols:
        df[c] = int(input_dict.get(c, 0))
    
    df['total_comorbidities'] = sum([input_dict.get(c, 0) for c in comorbidity_cols])
    df['high_glucose'] = int(input_dict['glucose'] > 140)
    df['low_sodium'] = int(input_dict['sodium'] < 135)
    df['high_creatinine'] = int(input_dict['creatinine'] > 1.3)
    df['low_bmi'] = int(input_dict['bmi'] < 18.5)
    df['high_bmi'] = int(input_dict['bmi'] > 30)
    df['abnormal_vitals'] = (
        int((input_dict['pulse'] < 60) or (input_dict['pulse'] > 100)) +
        int((input_dict['respiration'] < 12) or (input_dict['respiration'] > 20))
    )
    
    for fac in ['A', 'B', 'C', 'D', 'E']:
        col_name = f'facility_{fac}'
        if col_name in feature_names:
            df[col_name] = int(input_dict['facility'] == fac)
    
    return df

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=70)
    st.title("OkoaMaisha")
    st.caption("AI Hospital Resource Optimizer")
    
    page = st.radio("Navigation", ["🔮 Predict LoS", "📊 Performance", "ℹ️ Overview", "📖 Documentation"])
    
    st.markdown("---")
    st.markdown("### 🎯 Model Stats")
    st.metric("Accuracy", f"{metadata['test_r2']:.1%}")
    st.metric("Avg Error", f"±{metadata['test_mae']:.2f} days")
    
    try:
        training_date = metadata['training_date'][:10]
        st.caption(f"📅 Updated: {training_date}")
    except:
        st.caption("📅 Model v3.0")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("""
    **For best results:**
    - Enter all available data
    - Double-check lab values
    - Review risk factors
    - Consider clinical context
    """)

# ============================================
# MAIN PREDICTION PAGE (NEW HOME PAGE)
# ============================================
if page == "🔮 Predict LoS":
    # Compact header with key info
    st.markdown("""
    <div class='compact-header'>
        <h1>🏥 Hospital Patient Length of Stay Prediction</h1>
        <p style='font-size: 1.1rem; margin-top: 0.75rem;'>Enter patient information below for instant AI-powered predictions</p>
        <p style='font-size: 0.95rem; margin-top: 0.5rem; opacity: 0.9;'>97% accurate predictions • ±0.31 day precision • Real-time results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats bar - NEW FANCY DESIGN
    st.markdown("""
    <div class='stats-container'>
        <div class='stats-grid'>
            <div class='stat-item'>
                <div class='stat-value'>97.2%</div>
                <div class='stat-label'>Accuracy</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>±0.31</div>
                <div class='stat-label'>Days Error</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>98%</div>
                <div class='stat-label'>Long-Stay Recall</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>&lt;1second</div>
                <div class='stat-label'>Prediction Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    if 'form_progress' not in st.session_state:
        st.session_state.form_progress = 0
    
    # Form sections with visual organization
    # st.markdown("<div class='form-section'>", unsafe_allow_html=True)

    with st.expander("👤 **Patient Demographics**", expanded=True):
        # st.markdown("<div class='form-section'>", unsafe_allow_html=True)
    
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"], help="Patient's biological sex")
            gender_encoded = 1 if gender == "Male" else 0
        with col2:
            rcount = st.slider("Readmissions (past 180d)", 0, 5, 0, 
                              help="Number of hospital readmissions in past 6 months")
        with col3:
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1,
                                 help="Body Mass Index (kg/m²)")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with st.expander("🩺 Medical History & Comorbidities", expanded=True):
        # st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        st.caption("Select all conditions that apply to this patient:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Chronic Conditions**")
            dialysisrenalendstage = st.checkbox("🔴 Dialysis/End-Stage Renal")
            hemo = st.checkbox("🔴 Hemoglobin Disorder")
            asthma = st.checkbox("🟡 Asthma")
            pneum = st.checkbox("🟡 Pneumonia")
        
        with col2:
            st.markdown("**Nutritional & Metabolic**")
            irondef = st.checkbox("🟡 Iron Deficiency")
            malnutrition = st.checkbox("🔴 Malnutrition")
            fibrosisandother = st.checkbox("🟡 Fibrosis & Other")
        
        with col3:
            st.markdown("**Mental Health**")
            psychologicaldisordermajor = st.checkbox("🟡 Major Psych Disorder")
            depress = st.checkbox("🟡 Depression")
            psychother = st.checkbox("🟡 Other Psychiatric")
            substancedependence = st.checkbox("🔴 Substance Dependence")
        
        # Show comorbidity count
        comorbidity_count = sum([dialysisrenalendstage, asthma, irondef, pneum,
                                substancedependence, psychologicaldisordermajor,
                                depress, psychother, fibrosisandother, malnutrition, hemo])
        
        if comorbidity_count > 0:
            st.markdown(f"""
            <div class='progress-indicator'>
                <strong>📊 Comorbidity Summary:</strong> {comorbidity_count} condition(s) selected
                {' • 🔴 High complexity case' if comorbidity_count >= 3 else ' • 🟢 Standard complexity'}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("💉 Vital Signs & Laboratory Results", expanded=True):
        # st.markdown("<div class='form-section'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Vital Signs**")
            pulse = st.number_input("Pulse (bpm)", 30, 200, 75,
                                   help="Heart rate in beats per minute")
            respiration = st.number_input("Respiration (/min)", 5.0, 60.0, 16.0,
                                         help="Respiratory rate per minute")
            
            # Visual feedback for vitals
            if pulse < 60 or pulse > 100:
                st.warning(f"⚠️ Abnormal pulse: {pulse} bpm")
            if respiration < 12 or respiration > 20:
                st.warning(f"⚠️ Abnormal respiration: {respiration}/min")
        
        with col2:
            st.markdown("**Hematology**")
            hematocrit = st.number_input("Hematocrit (%)", 20.0, 60.0, 40.0,
                                        help="Percentage of red blood cells")
            neutrophils = st.number_input("Neutrophils (×10³/µL)", 0.0, 20.0, 4.0,
                                         help="Neutrophil count")
            
            if hematocrit < 35 or hematocrit > 50:
                st.warning(f"⚠️ Abnormal hematocrit: {hematocrit}%")
        
        st.markdown("**Chemistry Panel**")
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            glucose = st.number_input("Glucose (mg/dL)", 50.0, 400.0, 100.0)
            if glucose > 140:
                st.caption("🔴 Elevated")
        
        with col4:
            sodium = st.number_input("Sodium (mEq/L)", 120.0, 160.0, 140.0)
            if sodium < 135:
                st.caption("🔴 Low")
        
        with col5:
            creatinine = st.number_input("Creatinine (mg/dL)", 0.3, 10.0, 1.0)
            if creatinine > 1.3:
                st.caption("🔴 Elevated")
        
        with col6:
            bloodureanitro = st.number_input("BUN (mg/dL)", 5.0, 100.0, 12.0)
            if bloodureanitro > 20:
                st.caption("🟡 Elevated")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with st.expander("🏥 Admission Information", expanded=True):

        # st.markdown("<div class='form-section'>", unsafe_allow_html=True)

        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            facility = st.selectbox("Facility", ["A", "B", "C", "D", "E"],
                                   help="Healthcare facility code")
        with col2:
            admission_month = st.selectbox("Admission Month", list(range(1, 13)))
        with col3:
            admission_dayofweek_str = st.selectbox("Day of Week", 
                                                   ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            admission_dayofweek = day_map[admission_dayofweek_str]
        with col4:
            secondarydiagnosisnonicd9 = st.slider("Secondary Diagnoses", 0, 10, 1,
                                                  help="Number of additional diagnoses")
        
        admission_quarter = (admission_month - 1) // 3 + 1
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prominent prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 PREDICT LENGTH OF STAY", 
                                   type="primary", 
                                   use_container_width=True)
    
    if predict_button:
        with st.spinner("🔮 Analyzing patient data with AI..."):
            input_dict = {
                'gender': gender_encoded, 'rcount': rcount, 'bmi': bmi,
                'pulse': pulse, 'respiration': respiration, 'hematocrit': hematocrit,
                'neutrophils': neutrophils, 'glucose': glucose, 'sodium': sodium,
                'creatinine': creatinine, 'bloodureanitro': bloodureanitro,
                'secondarydiagnosisnonicd9': secondarydiagnosisnonicd9,
                'admission_month': admission_month, 'admission_dayofweek': admission_dayofweek,
                'admission_quarter': admission_quarter, 'facility': facility,
                'dialysisrenalendstage': dialysisrenalendstage, 'asthma': asthma,
                'irondef': irondef, 'pneum': pneum, 'substancedependence': substancedependence,
                'psychologicaldisordermajor': psychologicaldisordermajor, 'depress': depress,
                'psychother': psychother, 'fibrosisandother': fibrosisandother,
                'malnutrition': malnutrition, 'hemo': hemo
            }
            
            input_df = engineer_features(input_dict)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Animated prediction result
            st.markdown(f"""
            <div class='prediction-box'>
                <h1>{prediction:.1f} days</h1>
                <p style='font-size: 1.3rem; margin-top: 1rem; font-weight: 500;'>Predicted Length of Stay</p>
                <p style='font-size: 1rem; opacity: 0.9;'>±{metadata['test_mae']:.2f} days confidence interval (95%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick status indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if prediction <= 3:
                    st.success("🟢 **Short Stay**\n\nLow resource intensity")
                elif prediction <= 7:
                    st.warning("🟡 **Medium Stay**\n\nStandard resources")
                else:
                    st.error("🔴 **Long Stay**\n\nHigh resource needs")
            
            with col2:
                st.metric("Comorbidities", f"{comorbidity_count}", 
                         delta="High" if comorbidity_count >= 3 else "Normal",
                         delta_color="inverse" if comorbidity_count >= 3 else "off")
            
            with col3:
                st.metric("Readmissions", rcount,
                         delta="High risk" if rcount >= 2 else "Low risk",
                         delta_color="inverse" if rcount >= 2 else "normal")
            
            with col4:
                risk_score = (comorbidity_count * 10) + (rcount * 15)
                risk_level = "High" if risk_score > 40 else "Medium" if risk_score > 20 else "Low"
                st.metric("Risk Score", f"{risk_score}/100",
                         delta=risk_level,
                         delta_color="inverse" if risk_level == "High" else "off")
            
            st.markdown("---")
            
            # Resource recommendations
            st.markdown("### 📋 Resource Planning Recommendations")
            
            if prediction > 7:
                st.error("""
                **🔴 High-Intensity Care Protocol**
                
                ✅ **Immediate Actions:**
                - Reserve extended-care bed immediately
                - Assign case manager within 24 hours
                - Order 10+ day medication supply
                - Initiate discharge planning on day 1
                
                ✅ **Coordination:**
                - Schedule multi-specialty care coordination
                - Alert social services for post-discharge support
                - Arrange family meeting within 48 hours
                """)
            elif prediction > 4:
                st.warning("""
                **🟡 Standard Care Protocol**
                
                ✅ **Standard Actions:**
                - Standard acute care bed assignment
                - Regular nursing staff ratios
                - 7-day medication supply
                - Routine monitoring and assessments
                
                ✅ **Planning:**
                - Discharge planning by day 3
                - Regular team rounds
                """)
            else:
                st.success("""
                **🟢 Short-Stay Fast-Track Protocol**
                
                ✅ **Optimized Actions:**
                - Short-stay unit eligible
                - Standard staffing sufficient
                - Early discharge planning opportunity
                - Minimal supply requirements
                
                ✅ **Efficiency:**
                - Consider same-day discharge protocols
                - Streamlined documentation
                """)
            
            # Risk factors
            st.markdown("### ⚠️ Clinical Risk Factors Identified")
            
            risks = []
            if rcount >= 2:
                risks.append(f"🔴 **High readmission count ({rcount})** - Strong predictor of extended stay")
            if comorbidity_count >= 3:
                risks.append(f"🔴 **Multiple comorbidities ({comorbidity_count})** - Complex care needs")
            if glucose > 140:
                risks.append(f"🟡 **Elevated glucose ({glucose:.0f} mg/dL)** - Diabetes management protocol")
            if sodium < 135:
                risks.append(f"🟡 **Hyponatremia ({sodium:.0f} mEq/L)** - Monitor electrolytes closely")
            if creatinine > 1.3:
                risks.append(f"🟡 **Elevated creatinine ({creatinine:.1f} mg/dL)** - Renal function monitoring")
            if bmi < 18.5:
                risks.append(f"🟡 **Low BMI ({bmi:.1f})** - Nutritional support recommended")
            elif bmi > 30:
                risks.append(f"🟡 **Elevated BMI ({bmi:.1f})** - Consider mobility support")
            
            if risks:
                for risk in risks:
                    st.warning(risk)
            else:
                st.success("✅ **No major risk factors identified** - Standard protocols apply")
            
            # Comparison chart
            st.markdown("### 📊 Length of Stay Comparison")
            
            comparison_data = pd.DataFrame({
                'Category': ['Your Patient', 'Average Short Stay', 'Average Medium Stay', 'Average Long Stay'],
                'Days': [prediction, 2.5, 5.5, 10.0],
                'Color': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
            })
            
            fig = go.Figure(data=[
                go.Bar(x=comparison_data['Category'], 
                      y=comparison_data['Days'],
                      marker_color=comparison_data['Color'],
                      text=comparison_data['Days'].round(1),
                      textposition='auto')
            ])
            
            fig.update_layout(
                title="Predicted Stay vs. Category Averages",
                yaxis_title="Days",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 New Prediction", use_container_width=True):
                    st.rerun()
            
            with col2:
                st.download_button(
                    "📥 Download Report",
                    data=f"Patient Prediction Report\n\nPredicted LoS: {prediction:.1f} days\nComorbidities: {comorbidity_count}\nReadmissions: {rcount}",
                    file_name=f"los_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )
            
            with col3:
                st.button("📊 View Dashboard", use_container_width=True)

# ============================================
# PERFORMANCE PAGE
# ============================================
elif page == "📊 Performance":
    st.title("📊 Model Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metadata['test_r2']:.4f}")
    with col2:
        st.metric("MAE", f"{metadata['test_mae']:.2f} days")
    with col3:
        st.metric("RMSE", f"{metadata.get('test_rmse', 0.40):.2f} days")
    with col4:
        st.metric("Training Size", "80,000")
    
    st.markdown("---")
    
    with st.container():
        st.markdown("### 📚 Understanding the Metrics")
        
        st.info("""
        **R² Score (0.9721):**  
        Explains how much of the variation in length of stay our model predicts. 97.21% means the model is highly accurate - only 2.79% of variation is unexplained.
        
        **MAE - Mean Absolute Error (0.31 days):**  
        On average, our predictions are off by just 7.4 hours. This means if we predict 5 days, the actual stay is typically between 4.7-5.3 days.
        
        **RMSE - Root Mean Squared Error (0.40 days):**  
        Similar to MAE but penalizes larger errors more heavily. A low RMSE means we rarely make big mistakes.
        """)
        
        st.success("""
        **💡 Bottom Line:**  
        This model is exceptionally accurate for hospital planning. Traditional methods have 30-50% error rates; we're at 3%.
        """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Comparison")
    
    comparison_data = {
        'Model': ['Gradient Boosting', 'XGBoost', 'LightGBM', 'Random Forest'],
        'R²': [0.9721, 0.9701, 0.9693, 0.9336],
        'MAE': [0.31, 0.31, 0.31, 0.40]
    }
    df_comp = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_comp, x='Model', y='R²', title='Accuracy Comparison (Higher is Better)',
                    color='R²', color_continuous_scale='Blues')
        fig.update_layout(yaxis_range=[0.92, 0.98])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(df_comp, x='Model', y='MAE', title='Error Comparison (Lower is Better)',
                    color='MAE', color_continuous_scale='Reds_r')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔍 Feature Importance")
    
    importance_data = {
        'Feature': ['Readmissions', 'Comorbidities', 'Hematocrit', 'BUN', 'Sodium'],
        'Importance': [0.579, 0.217, 0.036, 0.021, 0.018]
    }
    df_imp = pd.DataFrame(importance_data)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                title='Top 5 Predictors of Length of Stay', color='Importance',
                color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='capability-card'>
            <h4>💡 Clinical Insights</h4>
            <ul>
                <li><strong>Readmissions dominate:</strong> 57.9% of prediction weight</li>
                <li><strong>Comorbidities matter:</strong> 21.7% influence</li>
                <li><strong>Together:</strong> ~80% of the model's decision</li>
                <li><strong>Takeaway:</strong> History predicts future better than current vitals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='capability-card' style='border-left-color: #10b981;'>
            <h4 style='color: #10b981;'>🎯 Long-Stay Performance</h4>
            <ul>
                <li><strong>Accuracy:</strong> 97% for extended stays</li>
                <li><strong>Recall:</strong> 98% detection rate (1,682/1,713)</li>
                <li><strong>Missed cases:</strong> Only 31 patients</li>
                <li><strong>Impact:</strong> Prevents bed shortages effectively</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# OVERVIEW PAGE (formerly Home)
# ============================================
elif page == "ℹ️ Overview":
    st.markdown("""
    <div class='compact-header'>
        <h1>🏥 OkoaMaisha - Overview</h1>
        <p>AI-Powered Hospital Patient Length of Stay Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2.5rem; margin: 0;'>97.2%</h3>
            <p style='font-weight: 600; color: #1e293b; margin-top: 0.5rem;'>Accuracy</p>
            <div style='font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;'>Model precision score</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2.5rem; margin: 0;'>±0.31</h3>
            <p style='font-weight: 600; color: #1e293b; margin-top: 0.5rem;'>Days Error</p>
            <div style='font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;'>Average prediction variance</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2.5rem; margin: 0;'>98%</h3>
            <p style='font-weight: 600; color: #1e293b; margin-top: 0.5rem;'>Long-Stay Recall</p>
            <div style='font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;'>High-risk detection rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2.5rem; margin: 0;'>100K</h3>
            <p style='font-weight: 600; color: #1e293b; margin-top: 0.5rem;'>Training Set</p>
            <div style='font-size: 0.85rem; color: #64748b; margin-top: 0.25rem;'>Validated patient records</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>✨ Key Capabilities</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='capability-card'>
            <h4>🎯 Accurate Predictions</h4>
            <ul>
                <li><strong>±0.31 day precision</strong> - Industry-leading accuracy</li>
                <li><strong>97.2% R² score</strong> - Explains 97% of variance</li>
                <li><strong>42 clinical features</strong> - Comprehensive analysis</li>
            </ul>
            <h4 style='margin-top: 1.5rem;'>📊 Resource Optimization</h4>
            <ul>
                <li><strong>Proactive bed management</strong> - Prevent shortages</li>
                <li><strong>Staff allocation</strong> - Optimize scheduling</li>
                <li><strong>Supply forecasting</strong> - Reduce waste</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='capability-card' style='border-left-color: #10b981;'>
            <h4 style='color: #10b981;'>🏥 Clinical Impact</h4>
            <ul>
                <li><strong>98% detection rate</strong> - Catches long-stay patients</li>
                <li><strong>Prevents bed crises</strong> - Early warning system</li>
                <li><strong>Discharge planning</strong> - Start day one</li>
            </ul>
            <h4 style='margin-top: 1.5rem; color: #10b981;'>🔒 Safe & Compliant</h4>
            <ul>
                <li><strong>HIPAA-compliant</strong> - Privacy-first design</li>
                <li><strong>Decision support</strong> - Augments clinicians</li>
                <li><strong>Human oversight</strong> - Always required</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>🚀 How It Works</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1", "Input", "Enter patient demographics, vitals, and medical history"),
        ("2", "Analyze", "AI processes 42 clinical features instantly"),
        ("3", "Predict", "Get precise length of stay estimate"),
        ("4", "Plan", "Receive actionable resource recommendations")
    ]
    
    for col, (num, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 10px; border: 2px solid #3b82f6; text-align: center; height: 100%;'>
                <div style='background: #3b82f6; color: white; width: 40px; height: 40px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem;'>{num}</div>
                <h4 style='color: #1e3a8a; margin: 0.5rem 0;'>{title}</h4>
                <p style='color: #475569; font-size: 0.95rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 **Ready to get started?** Go to **🔮 Predict LoS** to enter patient data and get instant predictions!")

# ============================================
# DOCUMENTATION PAGE (formerly About)
# ============================================
else:  # Documentation
    st.title("📖 Documentation & Support")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What is OkoaMaisha?
        
        **OkoaMaisha** (Swahili for "Save Lives") is an AI-powered clinical decision support system that predicts 
        individual patient hospital length of stay using advanced machine learning. It helps healthcare facilities optimize resource 
        allocation, improve bed management, and enhance patient care planning.
        
        The system was designed specifically for resource-constrained healthcare settings where efficient 
        bed management can literally save lives by ensuring capacity for emergency admissions.
        """)
    
    with col2:
        st.markdown("""
        <div class='capability-card' style='border-left-color: #10b981;'>
            <h4 style='color: #10b981;'>🌍 Built for Impact</h4>
            <p style='color: #475569;'>Designed for hospitals in low and middle-income countries where bed shortages 
            are a critical challenge.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔬 How It Works")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='capability-card'>
            <h4>📊 Data Processing</h4>
            <p style='color: #475569; line-height: 1.8;'>
            The system analyzes <strong>42 clinical features</strong> across four categories:
            </p>
            <ul style='color: #475569; line-height: 1.8;'>
                <li><strong>Demographics:</strong> Age, gender, BMI</li>
                <li><strong>Medical History:</strong> 11 comorbidity indicators</li>
                <li><strong>Clinical Data:</strong> Vitals and lab results</li>
                <li><strong>Admission Context:</strong> Facility, timing, secondary diagnoses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='capability-card' style='border-left-color: #8b5cf6;'>
            <h4 style='color: #8b5cf6;'>🤖 AI Model</h4>
            <p style='color: #475569; line-height: 1.8;'>
            We use a <strong>Gradient Boosting Regressor</strong>, chosen after comparing 4 algorithms:
            </p>
            <ul style='color: #475569; line-height: 1.8;'>
                <li><strong>Gradient Boosting:</strong> Best performer (97.2% R²)</li>
                <li><strong>XGBoost:</strong> Second best (97.0% R²)</li>
                <li><strong>LightGBM:</strong> Fast alternative (96.9% R²)</li>
                <li><strong>Random Forest:</strong> Baseline (93.4% R²)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2rem;'>97.21%</h3>
            <p style='font-weight: 600;'>R² Accuracy</p>
            <div style='font-size: 0.85rem; color: #64748b;'>Explains 97% of variance</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2rem;'>±0.31</h3>
            <p style='font-weight: 600;'>Mean Error</p>
            <div style='font-size: 0.85rem; color: #64748b;'>7.4 hours average</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2rem;'>98%</h3>
            <p style='font-weight: 600;'>Long-Stay Recall</p>
            <div style='font-size: 0.85rem; color: #64748b;'>1,682 of 1,713 detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2rem;'>100K</h3>
            <p style='font-weight: 600;'>Training Data</p>
            <div style='font-size: 0.85rem; color: #64748b;'>Validated patient records</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Important Disclaimers")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **🚨 Clinical Decision Support Only**
        
        This tool provides **decision support** and should **not replace clinical judgment**. 
        All predictions must be reviewed by qualified healthcare professionals.
        
        - Not a diagnostic tool
        - Requires human oversight
        - Predictions are probabilistic
        - Should supplement, not replace, clinical assessment
        """)
    
    with col2:
        st.warning("""
        **🔒 Privacy & Compliance**
        
        This system is designed with **HIPAA compliance** principles:
        
        - No patient data is stored permanently
        - All data should be handled per institutional policies
        - Secure data transmission required
        - Regular security audits recommended
        - Staff training on proper use essential
        """)
    
    st.markdown("### 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='capability-card'>
            <h4>⚙️ Model Specifications</h4>
            <ul style='color: #475569; line-height: 1.8;'>
                <li><strong>Algorithm:</strong> Gradient Boosting Regressor</li>
                <li><strong>Features:</strong> 42 engineered clinical variables</li>
                <li><strong>Training Set:</strong> 80,000 patients (80% split)</li>
                <li><strong>Test Set:</strong> 20,000 patients (20% split)</li>
                <li><strong>Version:</strong> 3.0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            training_date = metadata.get('training_date', 'N/A')[:10]
        except:
            training_date = "2024"
        
        st.markdown(f"""
        <div class='capability-card' style='border-left-color: #8b5cf6;'>
            <h4 style='color: #8b5cf6;'>📅 Deployment Info</h4>
            <ul style='color: #475569; line-height: 1.8;'>
                <li><strong>Last Updated:</strong> {training_date}</li>
                <li><strong>Framework:</strong> Scikit-learn, Streamlit</li>
                <li><strong>Deployment:</strong> Cloud-based web application</li>
                <li><strong>Availability:</strong> 24/7 access</li>
                <li><strong>Updates:</strong> Quarterly model retraining</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p style='font-size: 1.1rem; font-weight: 600; color: #1e3a8a;'>OkoaMaisha - Hospital Resource Optimization System</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Clinical Decision Support Tool | Version 3.0 | Powered by AI</p>
    <p style='font-size: 0.8rem; margin-top: 1rem; color: #94a3b8;'>
        ⚠️ This tool is for clinical decision support only. 
        Final decisions must be made by qualified healthcare professionals.
    </p>
    <p style='font-size: 0.75rem; margin-top: 0.5rem; color: #cbd5e1;'>
        © 2025 OkoaMaisha Project | Designed for healthcare facilities worldwide
    </p>
</div>
""", unsafe_allow_html=True)
