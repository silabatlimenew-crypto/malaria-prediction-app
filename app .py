import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Malaria Predictor", page_icon="🦟", layout="centered")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('best_malaria_model.pkl')

model = load_model()

# App UI Header
st.title("🦟 Malaria Prevalence Trend Predictor")
st.write("Enter historical malaria prevalence data below to predict the **2020 prevalence trend category** using your trained Random Forest model.")

# Input fields for the 4 historical features
st.subheader("Historical Data Input")
p_2000 = st.number_input("Malaria Prevalence in 2000", value=12.5, format="%.2f")
p_2005 = st.number_input("Malaria Prevalence in 2005", value=18.0, format="%.2f")
p_2010 = st.number_input("Malaria Prevalence in 2010", value=9.4, format="%.2f")
p_2015 = st.number_input("Malaria Prevalence in 2015", value=4.2, format="%.2f")

# Prediction button
if st.button("Predict 2020 Trend", type="primary"):
    input_df = pd.DataFrame(
        [[p_2000, p_2005, p_2010, p_2015]],
        columns=['malaria_prevalence_2000', 'malaria_prevalence_2005', 'malaria_prevalence_2010', 'malaria_prevalence_2015']
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result_label = "High Prevalence Trend" if prediction == 1 else "Low Prevalence Trend"
    confidence = probability[prediction] * 100

    st.divider()
    st.subheader("Prediction Results")
    if prediction == 1:
        st.error(f"**Result:** {result_label}")
    else:
        st.success(f"**Result:** {result_label}")

    st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
