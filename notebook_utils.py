import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests

def load_model():
    model_path = "churn_model.pkl"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1Wq0R1mK38ZTpqKHEQYvf-7QT-2VLd5yG"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return joblib.load(model_path)


# Preprocessing (must match training)
def preprocess_input(data):
    data = data.copy()

    # Feature engineering (same as training)
    data["Spend per Month"] = data["Total Spend"] / (data["Tenure"] + 1)
    data["Support Call Ratio"] = data["Support Calls"] / (data["Tenure"] + 1)

    # Age Group
    if "Age Group" not in data.columns:
        data["Age Group"] = pd.cut(data["Age"], 
                                   bins=[17, 25, 35, 45, 55, 65, 100], 
                                   labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
                                   right=True)

    # Categorical and numerical columns
    categorical_cols = ["Gender", "Subscription Type", "Contract Length", "Age Group"]
    numerical_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", 
                      "Payment Delay", "Total Spend", "Last Interaction", 
                      "Spend per Month", "Support Call Ratio"]

    # Encode categoricals (same mapping as training)
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Scale numerical (NOTE: For production, load fitted scaler from training instead)
    data[numerical_cols] = StandardScaler().fit_transform(data[numerical_cols])

    # Final feature set
    return data[numerical_cols + categorical_cols]

# Predict
def make_prediction(model, data):
    return model.predict(data)
