# Telco Customer Churn Prediction

A machine learning solution to predict which telecom customers will churn and explain the reasons behind each prediction using SHAP interpretability.

## Problem Statement

Customer churn is a critical business challenge in the telecommunications industry. This project aims to:
- Predict which customers are likely to leave
- Provide explainable reasons for each prediction
- Enable targeted retention strategies based on customer-specific insights

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, containing 7,043 customer records with features including:
- Demographics (age, gender, senior citizen status)
- Services (phone, internet, streaming, security, technical support)
- Account information (tenure, contract type, billing method)
- Charges (monthly and total charges)

## Tech Stack

- **Programming Language**: Python 3.8+
- **Machine Learning Model**: LightGBM (Gradient Boosting Classifier)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Optional API**: FastAPI
- **Optional GenAI**: Large Language Models (LLMs)

## Getting Started

## What is SHAP?

SHAP (SHapley Additive exPlanations) provides interpretable explanations for machine learning predictions. For each customer, it shows which features push the prediction toward churn (risk factors) and which features reduce the risk (protective factors).

## Project Goals

- Build an accurate churn prediction model
- Generate SHAP waterfall explanations for customer cohorts
- Create a CSV file with churn scores for all customers
- Optional: Deploy as a REST API for real-time predictions

## Status

Initial setup and core infrastructure. Development in progress.
