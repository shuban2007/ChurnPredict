import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("AI‑Based Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn using a trained Machine Learning model.")

# ---------------- LOAD MODEL SAFELY ----------------
if not os.path.exists("churn_model.pkl"):
    st.error("❌ Model file not found. Please upload churn_model.pkl")
    st.stop()

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- LABEL ENCODING MAPS ----------------
# MUST match LabelEncoder alphabetical order used in training

gender_map = {"Female": 0, "Male": 1}
yes_no_map = {"No": 0, "Yes": 1}

multiple_lines_map = {
    "No": 0,
    "No phone service": 1,
    "Yes": 2
}

internet_service_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

service_map = {
    "No": 0,
    "No internet service": 1,
    "Yes": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

# ---------------- INPUT FORM ----------------
with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
    with col2:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    with col3:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    st.subheader("Online Services")

    col4, col5, col6 = st.columns(3)
    with col4:
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    with col5:
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    with col6:
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.subheader("Billing Information")

    col7, col8, col9 = st.columns(3)
    with col7:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    with col8:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
    with col9:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    submit = st.form_submit_button("Predict Churn")

# ---------------- PREDICTION ----------------
if submit:
    with st.spinner("Analyzing customer data..."):
        time.sleep(1)

        input_data = pd.DataFrame([[
            gender_map[gender],
            senior_citizen,
            yes_no_map[partner],
            yes_no_map[dependents],
            tenure,
            yes_no_map[phone_service],
            multiple_lines_map[multiple_lines],
            internet_service_map[internet_service],
            service_map[online_security],
            service_map[online_backup],
            service_map[device_protection],
            service_map[tech_support],
            service_map[streaming_tv],
            service_map[streaming_movies],
            contract_map[contract],
            yes_no_map[paperless_billing],
            payment_map[payment_method],
            monthly_charges,
            total_charges
        ]], columns=[
            'gender','SeniorCitizen','Partner','Dependents','tenure',
            'PhoneService','MultipleLines','InternetService','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
            'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
            'MonthlyCharges','TotalCharges'
        ])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to **CHURN**\n\nProbability: **{probability:.2%}**")
        else:
            st.success(f"✅ Customer is likely to **STAY**\n\nProbability: **{1 - probability:.2%}**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI‑Based Customer Churn Prediction | Internship Project")
