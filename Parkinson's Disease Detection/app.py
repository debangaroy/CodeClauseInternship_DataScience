import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
parkinson_df = pd.read_csv("O:/virtual_intern/CodeClause-Data Science/Parkinson's Disease Detection/parkinsons.csv")

# Data Preprocessing
x = parkinson_df.drop(["status", "name"], axis=1)  # Drop non-numeric 'name' column if present
y = parkinson_df["status"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

def predict_parkinsons(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_input_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_input_data)
    return prediction[0]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = predict_parkinsons(features)

    if prediction == 0:
        result = "The person doesn't have Parkinson's disease"
    else:
        result = "The person has Parkinson's disease"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
