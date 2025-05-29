import streamlit as st
import numpy as np
from model_wrapper import SklearnLikeLogitModel
import joblib

model = joblib.load("logistic_model.pkl")

# Title
st.title("🔍 Retirement Adequacy Checker")
st.write("Estimate the likelihood of having adequate retirement income based on your current savings behavior.")

# User Inputs
st.header("📋 Enter Your Details")
salary = st.number_input("Current Monthly Salary (KES)", min_value=0, value=50000, step=1000)
retirement_age = st.number_input("Expected Retirement Age", min_value=50, max_value=70, value=60)
current_age = st.number_input("Current Age", min_value=18, max_value=69, value=30)
years_of_service = retirement_age - current_age

contribution_rate = st.slider("Estimated Contribution Rate (% of Salary)", min_value=0.0, max_value=0.5, value=0.1)
attended_training = st.radio("Have you attended any financial literacy training?", ["Yes", "No"])
voluntary_contributions = st.radio("Do you make voluntary pension contributions?", ["Yes", "No"])

# Convert to binary
training_attended = 1 if attended_training == "Yes" else 0
additional_contributions = 1 if voluntary_contributions == "Yes" else 0

# Pension Estimation
monthly_contribution = salary * contribution_rate
estimated_monthly_pension = monthly_contribution * years_of_service * 0.03  # Simplified growth + annuity factor
replacement_ratio = estimated_monthly_pension / salary

# Load pretrained model (this assumes you have a trained logistic model saved as .pkl)
# Replace 'logistic_model.pkl' with your actual model file path
try:
    model = joblib.load("logistic_model.pkl")
    X_input = np.array([[training_attended, additional_contributions, replacement_ratio]])
    prob_adequate = model.predict_proba(X_input)[0][1]

    st.header("📊 Results")
    st.metric("Estimated Replacement Ratio", f"{replacement_ratio:.2%}")
    st.metric("Probability of Income Adequacy", f"{prob_adequate:.2%}")

    if prob_adequate >= 0.7:
        st.success("🎉 You are likely to have adequate retirement income.")
    elif prob_adequate >= 0.4:
        st.warning("⚠️ Your adequacy is moderate. Consider increasing contributions or training.")
    else:
        st.error("❗ You are at high risk of inadequate retirement income.")

    st.write("\n**Tips:**")
    st.markdown("- Increase your monthly contributions if possible")
    st.markdown("- Attend a certified financial literacy program")
    st.markdown("- Delay retirement if feasible to increase years of service")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'logistic_model.pkl' is in the directory.")