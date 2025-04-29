# 💳 Credit Card Fraud Detection with Machine Learning

This project aims to develop and deploy a Machine Learning model capable of identifying fraudulent credit card transactions using supervised learning techniques and handling imbalanced datasets.

## 🚀 Try the App Online

👉 [Click here to test the fraud detection model in real time](https://detector-de-fraudes-victorbelle.streamlit.app/)

Upload a `.csv` file containing credit card transactions to see which ones are flagged as fraud by the trained XGBoost model.

---

## 📊 Project Overview

- **Problem Type:** Binary Classification (Fraud vs. Legitimate)
- **Dataset:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud)
- **Imbalance:** <1% of the data represents frauds

---

## 🧠 Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**
2. **Feature Scaling with StandardScaler**
3. **Handling Class Imbalance** using:
   - `class_weight="balanced"` in models
   - Consideration of `scale_pos_weight` in XGBoost
4. **Model Training and Evaluation**
   - Random Forest
   - XGBoost ✅ (*Best performance*)
5. **Model Selection Based on:**
   - Recall, Precision, F1-Score
   - AUC-ROC
6. **Deployment with Streamlit**

---

## 📈 Results Summary

| Model         | Recall (Fraud) | Precision (Fraud) | F1-Score | AUC-ROC |
|---------------|----------------|-------------------|----------|---------|
| Random Forest | 0.71           | 0.97              | 0.82     | ~       |
| XGBoost       | **0.79**       | 0.88              | **0.83** | **0.9705** ✅ |

XGBoost was selected for deployment due to its higher recall and AUC-ROC, making it more suitable for detecting rare fraudulent transactions.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, scikit-learn, XGBoost
- joblib
- Streamlit (for deployment)
- GitHub + Streamlit Cloud

---

## 📁 Project Structure

```bash
credit-card-fraud-detection/
├── models/                      # Trained model and scaler
│   ├── xgb_fraud_model.joblib
│   └── scaler.joblib
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_EDA_creditcard_fraud.ipynb
│   ├── 02_Preprocessing_and_Modeling.ipynb
│   └── 03_Model_Comparison_and_Optimization.ipynb
├── streamlit_app.py           # Main app for deployment
├── requirements.txt
└── README.md
