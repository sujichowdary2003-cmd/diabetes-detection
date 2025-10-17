import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load or train model
@st.cache_resource
def load_model():
    if os.path.exists("diabetes_model.pkl"):
        model = joblib.load("diabetes_model.pkl")
        return model
    # Load dataset
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
    df = pd.read_csv(url)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    joblib.dump(model, "diabetes_model.pkl")
    return model

model = load_model()

st.title("Diabetes Detection AI")
st.write("Enter patient information to predict the likelihood of diabetes.")

# List of features
features = [
    ("Pregnancies", "Number of times pregnant", 0, 20, 1),
    ("Glucose", "Plasma glucose concentration", 0, 200, 120),
    ("BloodPressure", "Diastolic blood pressure (mm Hg)", 0, 140, 70),
    ("SkinThickness", "Triceps skin fold thickness (mm)", 0, 100, 20),
    ("Insulin", "2-Hour serum insulin (mu U/ml)", 0, 900, 80),
    ("BMI", "Body mass index (weight in kg/(height in m)^2)", 0.0, 70.0, 30.0),
    ("DiabetesPedigreeFunction", "Diabetes pedigree function", 0.0, 2.5, 0.5),
    ("Age", "Age (years)", 1, 120, 33)
]

user_input = []
for f in features:
    if isinstance(f[2], int) and isinstance(f[3], int):
        val = st.number_input(f"{f[0]} ({f[1]})", min_value=f[2], max_value=f[3], value=f[4])
    else:
        val = st.number_input(f"{f[0]} ({f[1]})", min_value=float(f[2]), max_value=float(f[3]), value=float(f[4]), format="%.2f")
    user_input.append(val)

if st.button("Predict Diabetes"):
    data = np.array(user_input).reshape(1, -1)
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    if pred == 1:
        st.error(f"High chance of Diabetes (Probability: {prob:.2f})")
    else:
        st.success(f"Low chance of Diabetes (Probability: {prob:.2f})")

st.markdown("---")
st.markdown("**Note:** This tool is for demonstration purposes only and not a substitute for medical advice.")
