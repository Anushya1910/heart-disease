import streamlit as st
import numpy as np
import joblib

st.title("Heart Disease Prediction")

# Load the model once
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.pkl")

model = load_model()

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250)
chol = st.number_input("Cholesterol", min_value=50, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown)", [0, 1, 2, 3])

# Prepare the input for prediction
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("High risk of heart disease.")
    else:
        st.success("Low risk of heart disease.")
