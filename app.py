import streamlit as st
import pandas as pd
from notebook_utils import load_model, preprocess_input, make_prediction

st.title("📊 Churn Prediction App")

# Load model
model = load_model()

# Input form
st.header("Enter Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Premium", "VIP"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
age = st.slider("Age", 18, 100)
tenure = st.slider("Tenure", 0, 72)
usage = st.slider("Usage Frequency", 0, 100)
support_calls = st.slider("Support Calls", 0, 20)
payment_delay = st.slider("Payment Delay (days)", 0, 30)
spend = st.number_input("Total Spend")
last_interaction = st.slider("Last Interaction (days)", 0, 365)

# Prepare data
data = pd.DataFrame({
    "Gender": [gender],
    "Subscription Type": [subscription_type],
    "Contract Length": [contract_length],
    "Age": [age],
    "Tenure": [tenure],
    "Usage Frequency": [usage],
    "Support Calls": [support_calls],
    "Payment Delay": [payment_delay],
    "Total Spend": [spend],
    "Last Interaction": [last_interaction],
})

# Prediction
if st.button("Predict"):
    processed = preprocess_input(data)
    prediction = make_prediction(model, processed)
    st.subheader("Prediction:")
    st.write("🔴 Churn" if prediction[0] == 1 else "🟢 Not Churn")
