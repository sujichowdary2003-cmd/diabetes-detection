import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@st.cache_resource
def load_model():
    df = pd.read_csv("Diabetes Dataset_Training Part.csv")
    X = df[['Preg', 'Glucose', 'BPressure', 'SThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df['Outcome']

    X[["Glucose","BPressure","SThickness","Insulin","BMI"]] = X[["Glucose","BPressure","SThickness","Insulin","BMI"]].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    model = RandomForestClassifier(random_state=42, max_depth=7)
    model.fit(X_scaled, y)
    return model, imputer, scaler

model, imputer, scaler = load_model()

st.title("🩺 Diabetes Detection App")
st.markdown("Predict whether a person has diabetes based on health indicators.")

Preg = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0.0, 200.0, 100.0)
BPressure = st.number_input("Blood Pressure", 0.0, 150.0, 70.0)
SThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
Insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_df = pd.DataFrame([[Preg, Glucose, BPressure, SThickness, Insulin, BMI, DPF, Age]],
                             columns=['Preg','Glucose','BPressure','SThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
    input_df[["Glucose","BPressure","SThickness","Insulin","BMI"]] = input_df[["Glucose","BPressure","SThickness","Insulin","BMI"]].replace(0, np.nan)
    X_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=input_df.columns)
    prediction = model.predict(X_scaled)[0]
    
    if prediction == 1:
        st.error("🚨 The person is having Diabetes.")
    else:
        st.success("✅ The person is not having Diabetes.")
