import pickle
import numpy as np
import os

#Correct Path opening

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '..', 'model', 'diabetes_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

#Program title
print("\nDiabetes Risk Prediction System\n")

#Inputs from the user
preg = int(input("Pregnancies: "))
glucose = int(input("Glucose Level: "))
bp = int(input("Blood Pressure: "))
skin = int(input("Skin Thickness: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

import pandas as pd

data = pd.DataFrame([{
    'Pregnancies': preg,
    'Glucose': glucose,
    'BloodPressure': bp,
    'SkinThickness': skin,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}])

# Predicting
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

print("\n Result:")

# Risk levels 
if probability < 0.3:
    print("✅ Low Risk")
elif probability < 0.7:
    print("⚠️ Moderate Risk")
else:
    print("🚨 High Risk")

print(f"Risk Probability: {probability*100:.2f}%")

# Personalized suggestions for user
print("\n Personalized Suggestions:")

if glucose > 140:
    print("- Reduce sugar intake")
if bmi > 25:
    print("- Maintain healthy weight")
if age > 45:
    print("- Regular health checkups")
if bp > 90:
    print("- Control blood pressure")

print("\nStay healthy! 💙") 