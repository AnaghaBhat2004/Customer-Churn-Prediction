#  Customer Churn Prediction

A web-based application that predicts whether a customer is likely to **churn** (leave a service) or stay, based on customer details like tenure, contract type, payment method, and service usage. Built using Python, Flask, and a machine learning model (Random Forest), with a responsive frontend.

---

## **Features**

- Predict customer churn based on multiple factors:
  - Tenure, Monthly Charges, Total Charges
  - Contract type, Internet service, Payment method
  - Online services like Security, Backup, Streaming, Tech support
- Interactive web interface with real-time predictions
- One-hot encoding and preprocessing handled automatically in backend
- REST API for predictions

---

## **Demo Input Example**

- Tenure: 2 months  
- Monthly Charges: $95.0  
- Total Charges: $190.0  
- Gender: Female  
- Senior Citizen: No  
- Partner: No  
- Dependents: No  
- Internet Service: Fiber optic  
- Contract: Month-to-month  
- Payment Method: Electronic check  

**Prediction:** ⚠️ Customer likely to CHURN

---

## **Tech Stack**

- **Backend:** Python, Flask, scikit-learn, Pandas, Joblib  
- **Frontend:** HTML, CSS, JavaScript  
- **Machine Learning:** Random Forest Classifier with preprocessing pipeline  
- **Other:** Flask-CORS for handling frontend requests

---

Customer-Churn-Prediction/
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── model_pipeline.joblib   # Trained model
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── frontend/
│   └── index.html
│
└── README.md

git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction/backend
