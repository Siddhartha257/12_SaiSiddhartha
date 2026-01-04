import joblib
import pandas as pd
import shap
import numpy as np

def predict_churn(input_data):
    """
    Loads the saved model, predicts the output, and calculates SHAP values.
    
    Args:
        input_data (pd.DataFrame): New data matching the training schema.
        
    Returns:
        dict: 'prediction', 'churn_probability', and 'shap_values'.
    """
    # 1. Load the Model
    # Ensure 'best_churn_model.pkl' exists from the training step
    model = joblib.load('best_churn_model.pkl')
    
    # 2. Predict
    # The pipeline automatically handles the preprocessor transformations
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1] # Probability of Class 1 (Churn)

    # 3. Calculate SHAP Values
    # We must manually transform the data to pass it to the underlying LightGBM explainer
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Transform input (impute/scale/encode) without SMOTE
    X_transformed = preprocessor.transform(input_data)
    
    # Initialize Explainer
    explainer = shap.TreeExplainer(classifier)
    shap_vals = explainer.shap_values(X_transformed)
    
    # LightGBM binary case often returns a list [class0_shap, class1_shap]. 
    # We return class 1 SHAP values if it's a list.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "shap_values": shap_vals
    }

# Example Usage
if __name__ == "__main__":
    # Dummy data structure based on standard Telco Churn dataset
    sample_data = {
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [70.0],
        'TotalCharges': ["840.0"]
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    try:
        results = predict_churn(df_sample)
        print(f"Prediction (0=No, 1=Yes): {results['prediction'][0]}")
        print(f"Churn Probability: {results['churn_probability'][0]:.2%}")
        print("SHAP Values Calculated.")
        print(results['shap_values'])
    except Exception as e:
        print(f"Error: {e}")