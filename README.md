# Telco Customer Churn Prediction with GenAI & FastAPI

A comprehensive machine learning solution designed to predict telecom customer churn. This project goes beyond simple prediction by integrating **Generative AI** to provide human-readable explanations for *why* a customer is at risk, served via a **FastAPI** backend.

## 🚀 Key Features

* **Recall-Optimized Modeling:** The primary goal is to minimize false negatives (missing a customer who is about to churn). The model is tuned to capture as many at-risk customers as possible.
* **Advanced Machine Learning:** Utilizes ensemble models (LightGBM, XGBoost, Random Forest) and Deep Learning (TensorFlow/Keras).
* **Imbalance Handling:** Implements **SMOTE** and **KMeansSMOTE** to handle class imbalance, ensuring the model doesn't just predict the majority class.
* **Explainable AI (XAI):** Uses **SHAP** (SHapley Additive exPlanations) to quantify feature contributions.
* **Generative AI Integration:** Integrated **Google Gemini Pro** to interpret SHAP values and generate personalized retention strategies for each customer.
* **REST API:** A robust **FastAPI** backend to serve predictions and explanations in real-time.

## 📂 Project Structure

```text
12_SaiSiddhartha/
├── backend/
│   ├── best_churn_model.pkl      # The serialized, trained ML model
│   ├── DataSet.csv               # Dataset used for training/inference
│   ├── model.ipynb               # Jupyter Notebook for EDA, training, and evaluation
│   ├── prediction.py             # Core logic for loading models and generating predictions
│   └── routes.py                 # FastAPI endpoints (API definition)
├── frontend/                     # Frontend source code (if applicable)
├── cross_model_valid...          # Scripts for cross-validation and model comparison
├── GUIDELINES.md                 # Project coding standards and guidelines
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Tech Stack

* **Core Frameworks:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost, TensorFlow/Keras
* **Data Processing:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **GenAI & LLMs:** Google Generative AI (Gemini)
* **Visualization:** Matplotlib, Seaborn, SHAP

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Siddhartha257/12_SaiSiddhartha.git
cd 12_SaiSiddhartha
```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

## 🏃‍♂️ How to Run

### 1. Start the FastAPI Backend

Navigate to the root directory and run the server using Uvicorn:

```bash
uvicorn backend.routes:app --reload
```

### 2. Access the API

* **API Root:** `http://127.0.0.1:8000`
* **Interactive Docs (Swagger UI):** `http://127.0.0.1:8000/docs`

## 🧠 Model Workflow

1. **Data Ingestion:** Load raw CSV data.
2. **Preprocessing:** One-hot encoding for categorical variables, scaling for numerical ones.
3. **Resampling:** Applied SMOTE to balance the dataset.
4. **Training:** Benchmarked multiple models; `LightGBM` was selected as the champion model.
5. **Inference:**
   * The API receives customer data.
   * Model predicts churn probability.
   * SHAP values are calculated to find top contributing factors.
   * GenAI reads these factors and writes a summary (e.g., "High risk due to month-to-month contract...").

## 📈 Results

* **Metric Focus:** Recall
* **Performance:** The model prioritizes identifying all potential churners to ensure no at-risk customer is ignored.
* **Key Drivers:** Analysis identified Contract Type, Monthly Charges, and Tenure as the strongest predictors of churn.



---



## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Siddhartha257/12_SaiSiddhartha/issues).

**Special thanks to the team:**
* **Sai Siddhartha** 
* **Chaitanya** 
* **Pranav Aditya Bongi** 


---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
