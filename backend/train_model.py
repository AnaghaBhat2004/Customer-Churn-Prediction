import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean 'TotalCharges' (convert blanks to numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn'].map({'Yes':1, 'No':0})  # Convert target to 0/1

# Identify categorical and numeric columns
categorical_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
numeric_cols = ['tenure','MonthlyCharges','TotalCharges']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# Full pipeline with classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])

# Train model
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, 'model_pipeline.joblib')

print("✅ Model trained and saved as 'model_pipeline.joblib'")
