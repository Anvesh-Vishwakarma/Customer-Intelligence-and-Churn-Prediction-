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

### Explainability

We can explain and understand the Random forest model using explainable AI modules such as Permutation Importance, Partial Dependence plots and Shap values.

1. Permutation Importance shows feature importance by randomly shuffling feature values and measuring how much it degrades our performance.

<p align="center">
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/eli51.png height=250 width=200>
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/eli52.png height=130 width=200> 
</p>

2. Partial dependence plot is used to see how churning probability changes across the range of particular feature. For example, in below graph of tenure group, the churn probability decreases at a higher rate if a person is in tenure group 2 compared to 1.

<p align="center">
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/pdp_tenure.png height=250 width=400>
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/pdp_contract.png height=250 width=400> 
</p>

<p align="center">
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/pdp_monthly_charges.png height=250 width=400>
<img src=https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/pdp_total_charges.png height=250 width=400> 
</p>

3. Shap values (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. In below plot we can see that why a particual customer's churning probability is less than baseline value and which features are causing them.

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/shap.png)

## Flask App

I saved the final tuned Random Forest model and deployed it using Flask web app. Flask is a micro web framework written in Python.  It is designed to make getting started quick and easy, with the ability to scale up to complex applications. I saved the shap value explainer tuned using random forest model to show shap plots in app. I have also utilized the cox-proportional hazard model to show survival curve and hazard curve, and to calculate expected customer lifetime value. 

The final app shows churning probability, gauge chart of how severe a customer is and shap values based on customer's data. The final app layout can be seen above.  







