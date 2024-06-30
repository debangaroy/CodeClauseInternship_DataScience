import numpy as np
import streamlit as st
import pickle

# Load the trained model and scaler
model_path = "O:/virtual_intern/CodeClause-Data Science/Heart Disease Risk Assessment/trained_heart_model.sav"
scaler_path = 'O:/virtual_intern/CodeClause-Data Science/Heart Disease Risk Assessment/scaler.sav'
load_model = pickle.load(open(model_path, 'rb'))
load_scaler = pickle.load(open(scaler_path, 'rb'))

# Prediction Function
def prediction_function(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_input_data = load_scaler.transform(input_data_reshaped)
    prediction = load_model.predict(std_input_data)
    if prediction[0] == 0:
        return "Low Risk of Heart Disease"
    else:
        return "High Risk of Heart Disease"

# Validate and convert input
def validate_input(value, field_name):
    try:
        return float(value)
    except ValueError:
        st.error(f"Invalid input for {field_name}. Please enter a valid number.")
        return None

# Streamlit app
def main():
    st.title("Heart Disease Risk Assessment")

    # User input parameters
    age = st.text_input("Age")
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cholesterol = st.text_input("Cholesterol")
    heart_rate = st.text_input("Heart Rate")
    diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
    family_history = st.selectbox("Family History", options=["No", "Yes"])
    smoking = st.selectbox("Smoking", options=["No", "Yes"])
    obesity = st.selectbox("Obesity", options=["No", "Yes"])
    alcohol_consumption = st.text_input("Alcohol Consumption")
    exercise_hours = st.text_input("Exercise Hours Per Week")
    diet = st.selectbox("Diet", options=["Unhealthy", "Average", "Healthy"])
    previous_heart_problems = st.selectbox("Previous Heart Problems", options=["No", "Yes"])
    medication_use = st.selectbox("Medication Use", options=["No", "Yes"])
    stress_level = st.text_input("Stress Level")
    sedentary_hours = st.text_input("Sedentary Hours Per Day")
    income = st.text_input("Income")
    bmi = st.text_input("BMI")
    triglycerides = st.text_input("Triglycerides")
    physical_activity_days = st.text_input("Physical Activity Days Per Week")
    sleep_hours = st.text_input("Sleep Hours Per Day")
    systolic_bp = st.text_input("Systolic BP")
    diastolic_bp = st.text_input("Diastolic BP")

    # Convert categorical inputs to numeric
    sex = 1 if sex == "Male" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0
    smoking = 1 if smoking == "Yes" else 0
    obesity = 1 if obesity == "Yes" else 0
    previous_heart_problems = 1 if previous_heart_problems == "Yes" else 0
    medication_use = 1 if medication_use == "Yes" else 0
    diet_map = {"Unhealthy": 0, "Average": 1, "Healthy": 2}
    diet = diet_map[diet]

    # Collecting user inputs with validation
    input_data = [
        validate_input(age, "Age"), sex, validate_input(cholesterol, "Cholesterol"), validate_input(heart_rate, "Heart Rate"), diabetes,
        family_history, smoking, obesity, validate_input(alcohol_consumption, "Alcohol Consumption"),
        validate_input(exercise_hours, "Exercise Hours Per Week"), diet, previous_heart_problems, medication_use,
        validate_input(stress_level, "Stress Level"), validate_input(sedentary_hours, "Sedentary Hours Per Day"), validate_input(income, "Income"), validate_input(bmi, "BMI"),
        validate_input(triglycerides, "Triglycerides"), validate_input(physical_activity_days, "Physical Activity Days Per Week"), validate_input(sleep_hours, "Sleep Hours Per Day"),
        validate_input(systolic_bp, "Systolic BP"), validate_input(diastolic_bp, "Diastolic BP")
    ]

    # Ensure all inputs are valid before making a prediction
    if None not in input_data:
        if st.button("Predict"):
            result = prediction_function(input_data)
            st.success(result)

if __name__ == "__main__":
    main()

#python -m streamlit run heart_Disease_Risk_Predict_Web_App.py
