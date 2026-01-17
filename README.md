## Customer Churn Prediction
I aim to implement a machine learning model to accurately predict if the customer will churn or not.

### Analysis

**Churn and Tenure Relationship:**

<p align="center">
<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/tenure-churn.png" width="600" height="300"/>
</p>

- As we can see the higher the tenure, the lesser the churn rate. This tells us that the customer becomes loyal with the tenure.

<br />

**Tenure Distrbution by Various Services:**

<p align="center">
<img src="https://github.com/Anvesh-Vishwakarma/Customer-Intelligence-and-Churn-Prediction-/blob/main/images/tenure_streamingmovies.png" width="340" height="250"/>
</p>

- When the customers are new they do not opt for various services and their churning rate is very high. This can be seen in above plot for Streaming Movies and this holds true for all various services.

<br />

**Internet Service By Contract Type:**

<p align="center">
<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/internetservice-contract.png" width="360" height="250"/>
</p>

- Many of the people of who opt for month-to-month Contract choose Fiber optic as Internet service and this is the reason for higher churn rate for fiber optic Internet service type.

<br />

**Payment method By Contract Type:**

<p align="center">
<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/payment-contract.png" width="500" height="250"/>
</p>

- People having month-to-month contract prefer paying by Electronic Check mostly or mailed check. The reason might be short subscription cancellation process compared to automatic payment.

<br />

**Monthly Charges:**

<p align="center">
<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/monthlycharges.png" width="300" height="220"/>
</p>

- As we can see the customers paying high monthly fees churn more.

<br />

### Modelling

For the modelling, I will use tress based Ensemble method as we do not have linearity in this classification problem. Also, we have a class imbalance of 1:3 and to combat it I will assign class weightage of 1:3 which means false negatives are 3 times costlier than false positives. I built a model on 80% of data and validated model on remaining 20% of data keeping in mind that I do not have data leakage.

The final model resulted in 0.85 F1 score and 0.71 ROC-AUC. The resulting plots can be seen below.

<p align="center">
<img src="https://github.com/Anvesh-Vishwakarma/Customer-Intelligence-and-Churn-Prediction-/blob/main/images/confusion_matrix.png" width="600" height="300"/>
</p>

## 📊 Business Impact & Model Explainability

### Why Explainability Matters
Predictive performance alone is not sufficient for real-world machine learning systems. This churn prediction model is designed to be **business-aware and explainable**, clearly translating model outputs into **measurable financial impact**. The goal is to enable proactive retention strategies that directly reduce customer churn and preserve recurring revenue.

---

## 🔍 Value Creation Pipeline

The system follows a transparent, explainable pipeline:

1. **Churn Risk Prediction**  
   A supervised machine learning classifier generates churn probabilities for each customer.

2. **High-Risk Customer Targeting**  
   Customers exceeding a defined churn probability threshold are selected for retention campaigns.

3. **Revenue Impact Estimation**  
   Business impact is quantified by estimating how many high-risk customers can be successfully retained and the revenue preserved as a result.

---

## 🧮 Mathematical Breakdown of Revenue Savings

### 1️. Churn Population Estimation
Let:
- `N` = total number of subscribers  
- `r_c` = churn rate  

Churners = N × r_c

For a 30,000-subscriber base:
        = 30,000 x 26.5
        
### 2️. Model Effectiveness (Recall on Churn Class)
* r_m = model recall
       CorrectlyIdentifiedChurners = 7,950 x 70
       
### 3️. Retention Campaign Success
* r_s = retention success rate
       RetainedCustomers = 5,565 x 70

### 4️. Revenue Preservation Calculation
Let:
* ARPU = average monthly revenue per customer
* T = revenue horizon (in months)
       AnnualRevenueperCustomer = 74.44 x 12 = 893.28
      TotalRevenueSaved = 3,896 x 893.28 ≈ 35Lakhsperyear
---

### 📌 Business Parameters

| Parameter | Description | Value |
|---------|------------|------|
| Total subscribers | Customer base size | 30,000 |
| Churn rate | Percentage of customers leaving | 26.5% |
| Model recall | Correctly identified churners | 70% |
| Retention success rate | Successfully retained churners | 70% |
| Average monthly revenue | Revenue per customer | ₹74.44 |
| Revenue horizon | Time window | 12 months |

---

3. Shap values (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. In below plot we can see that why a particual customer's churning probability is less than baseline value and which features are causing them.

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/shap.png)

## Streamlit App

I saved the final tuned Random Forest model and deployed it using Streamlit web app. Streamlit is a micro web framework written in Python.  It is designed to make getting started quick and easy, with the ability to scale up to complex applications. 

The final app shows churning probability.  

## How to run

### 1️. Clone the repository
```
https://github.com/Anvesh-Vishwakarma/Customer-Intelligence-and-Churn-Prediction-.git
```

### 2️. Create Virtual Enviornment
```
python -m venv venv
```
Activate virtual enviornment
```
venv/Scripts/activate  
```

### 3. Install dependencies
```

pip install -r requirements.txt
```

### 4. Run the FastAPI which is inside api folder in app.py
```
uvicorn app:app --relaod
```
### 5. Run the streamlit app in frontend.py

```
Streamlit run frontend.py
```

**NOTE: Make sure FastAPI running**











