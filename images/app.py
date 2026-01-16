from pydantic import BaseModel, Field, computed_field
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
from typing import Annotated, Literal
import pandas as pd

app = FastAPI()

with open("churn_prediction_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]          
feature_order = model_data["features"]

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

class input_data(BaseModel):

    gender: Annotated[Literal["Male","Female"],Field(...,description="Select the gender")]
    SeniorCitizen: Annotated[Literal["Yes","No"],Field(...,description="Is the customer a senior citizen?")]
    Partner: Annotated[Literal["Yes","No"],Field(...,description="Does the customer have a partner?")]
    Dependents: Annotated[Literal["Yes","No"],Field(...,description="Does the customer have dependents?")]
    tenure: Annotated[int,Field(...,description="Number of months the customer has been with the company")]
    PhoneService: Annotated[Literal["Yes","No"],Field(...,description="Does the customer have phone service?")]
    MultipleLines: Annotated[Literal["Yes","No","No phone service"],Field(...,description="Does the customer have multiple lines?")]
    InternetService: Annotated[Literal["DSL","Fiber optic","No"],Field(...,description="What type of internet service does the customer have?")]
    OnlineSecurity: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have online security?")]
    OnlineBackup: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have online backup?")]
    DeviceProtection: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have device protection?")]
    TechSupport: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have tech support?")]
    StreamingTV: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have streaming TV?")]
    StreamingMovies: Annotated[Literal["Yes","No","No internet service"],Field(...,description="Does the customer have streaming movies?")]
    Contract: Annotated[Literal["Month-to-month","One year","Two year"],Field(...,description="What is the contract type?")]
    PaperlessBilling: Annotated[Literal["Yes","No"],Field(...,description="Does the customer have paperless billing?")]
    PaymentMethod: Annotated[Literal["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],Field(...,description="What is the payment method?")]
    MonthlyCharges: Annotated[float,Field(...,description="Monthly charges")]
    TotalCharges: Annotated[float,Field(...,description="Total charges")]
    

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: input_data):

    df = pd.DataFrame([data.model_dump()])

    # Convert 'Yes'/'No' to 1/0 for senior_citizen
    df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    for col, le in encoder.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    prediction = model.predict(df)
    churn_prob = model.predict_proba(df)[:, 1][0]
    return {
        "churn_prediction": "Yes" if prediction[0] == 1 else "No",
        "churn_probability": churn_prob
    }
