import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Malaria Prevalence Trend Predictor")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('best_malaria_model.pkl')

model = load_model()

# App UI Header
st.title("🦟 Malaria Prevalence Trend Predictor")
st.write("Enter historical malaria prevalence data below:")

# Input fields for the 4 historical features
st.subheader("Historical Data Input")
p_2000 = st.number_input("Malaria Prevalence in 2000", value=0.0, step=0.1)
p_2005 = st.number_input("Malaria Prevalence in 2005", value=0.0, step=0.1)
p_2010 = st.number_input("Malaria Prevalence in 2010", value=0.0, step=0.1)
p_2015 = st.number_input("Malaria Prevalence in 2015", value=0.0, step=0.1)

# Prediction button
if st.button("Predict 2020 Trend", type="primary"):
    # Create input dataframe
    input_df = pd.DataFrame(
        [[p_2000, p_2005, p_2010, p_2015]],
        columns=['malaria_prevalence_2000', 'malaria_prevalence_2005', 'malaria_prevalence_2010', 'malaria_prevalence_2015']
    )
    
    # Get probabilities instead of hard prediction to adjust threshold if needed
    probabilities = model.predict_proba(input_df)[0]
    
    # Let's check probabilities: index 0 is Low/Normal, index 1 is High
    prob_high = probabilities[1]
    
    # Custom threshold (e.g., if probability of high is > 0.5, else low)
    prediction = 1 if prob_high >= 0.5 else 0
    result_label = "High Prevalence Trend" if prediction == 1 else "Low/Moderate Prevalence Trend"
    confidence = prob_high * 100 if prediction == 1 else (1 - prob_high) * 100

    st.divider()
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"**Result:** {result_label}")
    else:
        st.success(f"**Result:** {result_label}")
        
    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

