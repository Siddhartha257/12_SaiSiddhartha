import os
import io
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Import your prediction logic
from prediction import predict_churn




#GEMINI API KEY 

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load model at startup
try:
    model = joblib.load('best_churn_model.pkl')
except Exception as e:
    print(f"WARNING: Model file not found. Please run 'main_model.py' first. Error: {e}")
    model = None

app = FastAPI(title="Churn Prediction API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STORAGE ---
latest_prediction_storage = {}

# --- DATA MODELS ---
class CustomerProfile(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: str = "840.0" 

class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    shap_values: List[float]
    feature_names: List[str]



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same preprocessing steps as the training script.
    """
    df = df.copy()
    
    # Drop ID if present
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2. Handle TotalCharges (Convert to numeric, coerce errors to NaN, fill with 0)
    # Remove any empty spaces or strange characters
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # 3. Normalize Categorical Values (Match Training Logic)
    # The training script replaced 'No internet service' with 'No'
    replace_cols = [ 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies' ]
    
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')
            df[col] = df[col].replace('No phone service', 'No')

    return df

async def get_gemini_explanation(prompt: str):
    model_gen = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = await model_gen.generate_content_async(prompt)
    return response.text



#ENDPOINTS

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerProfile):
    global latest_prediction_storage
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Convert to DataFrame
        input_data = customer.model_dump()
        df = pd.DataFrame([input_data])
        
        # 2. Clean Data (Crucial step!)
        df_clean = clean_data(df)
        
        # 3. Predict
        pred = model.predict(df_clean)[0]
        prob = model.predict_proba(df_clean)[0][1]
        
        # 4. SHAP Values
        # Extract components from pipeline
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform using the pipeline's preprocessor
        X_transformed = preprocessor.transform(df_clean)
        
        # Calculate SHAP
        import shap
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_transformed)
        
        # Handle SHAP output format (LightGBM binary often returns list of arrays)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] # Class 1 (Churn)
        
        if hasattr(shap_vals, 'flatten'):
            shap_vals = shap_vals.flatten().tolist()
            
        feature_names = list(input_data.keys())

        # Store for Explainability
        latest_prediction_storage = {
            "churn_probability": prob,
            "shap_values": shap_vals,
            "feature_names": feature_names
        }

        return {
            "prediction": int(pred),
            "churn_probability": float(prob),
            "shap_values": shap_vals,
            "feature_names": feature_names
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to server console
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict_batch_analysis")
async def predict_batch_analysis(file: UploadFile = File(...)):
    """
    1. Process CSV
    2. Identify High-Risk group
    3. Calculate Aggregate SHAP (Key Drivers for the group)
    4. Get LLM Strategy Report
    5. Return JSON with data + report
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Read & Clean
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df_clean = clean_data(df)
        
        # 2. Predict
        probs = model.predict_proba(df_clean)[:, 1]
        preds = model.predict(df_clean)
        
        # 3. Attach results
        results_df = df.copy()
        results_df['Churn_Probability'] = probs
        results_df['Risk_Level'] = np.where(probs > 0.6, 'High', 'Low')
        
        # 4. Analyze High Risk Group for LLM
        high_risk_df = df_clean[probs > 0.6]
        churn_rate = (len(high_risk_df) / len(df)) * 100
        
        # Calculate approximate key drivers for the high-risk group
        # (Using mean values of key features to keep it fast/simple for LLM context)
        # For a more advanced version, we would aggregate SHAP values here.
        
        summary_stats = ""
        if not high_risk_df.empty:
            # Get top 3 categorical frequent values in high risk group
            cat_summary = []
            for col in ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']:
                if col in high_risk_df.columns:
                    top_val = high_risk_df[col].mode()[0]
                    cat_summary.append(f"{col}: {top_val}")
            
            avg_tenure = high_risk_df['tenure'].mean() if 'tenure' in high_risk_df.columns else 0
            avg_charge = high_risk_df['MonthlyCharges'].mean() if 'MonthlyCharges' in high_risk_df.columns else 0
            
            summary_stats = f"""
            - High Risk Customers: {len(high_risk_df)} ({churn_rate:.1f}% of total)
            - Average Tenure of Risk Group: {avg_tenure:.1f} months
            - Average Monthly Charge: ${avg_charge:.2f}
            - Common Patterns: {', '.join(cat_summary)}
            """
        else:
            summary_stats = "Great news! Very few high-risk customers detected."

        # Generate LLM Strategy Report
        prompt = f"""
        You are a generic Strategy Consultant. 
        I have analyzed a batch of {len(df)} customers. Here is the summary of the "At Risk" segment:
        
        {summary_stats}
        
        Based ONLY on these stats, provide a brief "Retention Strategy Report" with:
        1. **Executive Summary**: What is the main problem?
        2. **3 Actionable Strategies**: Specific things to do (e.g., if they are on Month-to-month, suggest 1-year discounts).
        
        Keep it professional, concise, and use markdown formatting.
        """
        
        # Call Gemini
        report = await get_gemini_explanation(prompt)
        
        # Return Data + Report (Convert DF to dict for JSON response)
        # We replace NaN with None for valid JSON
        return {
            "strategy_report": report,
            "data": results_df.replace({np.nan: None}).to_dict(orient="records")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_shap")
async def explain_shap():
    if not latest_prediction_storage:
        raise HTTPException(status_code=400, detail="No prediction found. Run /predict first.")

    try:
        data = latest_prediction_storage
        prob = data["churn_probability"]
        
        # Sort features by impact
        feats = list(zip(data["feature_names"], data["shap_values"]))
        feats.sort(key=lambda x: abs(x[1]), reverse=True)
        top_5 = feats[:5]
        
        prompt = f"""
        Act as a Retention Manager. Analyze this customer:
        Churn Risk: {prob:.1%} ({'High' if prob > 0.5 else 'Low'})
        
        Top Drivers:
        {chr(10).join([f"- {f[0]}: {f[1]:.3f}" for f in top_5])}
        
        Briefly explain why they might leave and suggest 1 retention action.
        """
        
        explanation = await get_gemini_explanation(prompt)
        return {"explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)