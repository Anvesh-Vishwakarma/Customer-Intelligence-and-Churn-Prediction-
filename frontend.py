import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

API_URL = "https://fastapi-churn-prediction.onrender.com"  

st.title("Churn Prediction")

with st.sidebar:
    st.header("📘 Model Information")
    st.write("**Model Used:** Random Forest Classifier")
    st.write("**Accuracy:** 84%")
    st.markdown("---")
    st.image("https://static.vecteezy.com/system/resources/previews/020/851/200/non_2x/forecasting-icon-design-free-vector.jpg")
    
    
st.subheader("🧾 Enter the customer details below to predict whether the customer will churn or not.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=29.85)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=29.85)


if st.button("Predict"):
    # Prepare input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200:
            st.success(f"Predicted Churn: **{result['churn_prediction']}**")
            st.info(f"Churn Probability: **{result['churn_probability']:.2f}**")
        
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")

st.markdown("---")
st.subheader(" Global Model Explainability")

data = [
    ["tenure", 0.245, "↓ Churn", "Long-term customers are less likely to churn"],
    ["Contract_Monthly", 0.192, "↑ Churn", "Month-to-month contracts increase churn risk"],
    ["MonthlyCharges", 0.168, "↑ Churn", "Higher monthly bills increase dissatisfaction"],
    ["TotalCharges", 0.112, "↓ Churn", "High lifetime value customers are retained"],
    ["OnlineSecurity_No", 0.089, "↑ Churn", "Lack of security services increases churn"],
    ["TechSupport_No", 0.074, "↑ Churn", "Poor technical support drives churn"],
    ["PaymentMethod_ElectronicCheck", 0.058, "↑ Churn", "Manual payments indicate churn risk"],
    ["InternetService_FiberOptic", 0.043, "↑ Churn", "Fiber plans are more price-sensitive"],
    ["SeniorCitizen", 0.012, "↑ Churn", "Senior customers churn slightly more"],
    ["PaperlessBilling", 0.007, "↑ Churn", "Low engagement customers churn more"]
]

df = pd.DataFrame(
    data,
    columns=["Feature", "Importance Score", "Impact", "Business Explanation"]
)

st.subheader("🔍 Model Explainability")
st.dataframe(df)

st.markdown("---")
st.subheader("📈 Model Performance (ROC Curve)")

st.info(
    "ROC curve shown using **synthetic probabilities** for demonstration.\n"
    "In production, compute using hold-out test data."
)
# Generate demo data
np.random.seed(42)
n = 1000
y_true = np.random.randint(0, 2, n)
y_prob = np.random.rand(n)  # Demo probabilities

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (Demo)')
ax.legend(loc="lower right")
st.pyplot(fig)



    
