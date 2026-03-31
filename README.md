# 🩺 AI-Based Diabetes Risk Predictor

## Overview
This project is a Command Line Interface (CLI) based machine learning system that predicts the risk of diabetes using health parameters. It also provides personalized suggestions based on the prediction.

## Features
- Logistic Regression model
- Risk classification (Low, Moderate, High)
- Probability-based prediction
- Personalized health recommendations
- CLI-based execution

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Project Structure
- data/ → dataset
- model/ → training script + trained model
- src/ → prediction system

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python model/train_model.py

3. Run prediction:
python src/predict.py

## Results
The model achieved an accuracy of approximately 74.68%.

## Future Scope
- Convert into web application
- Improve accuracy using advanced models
- Add real-time data integration

## Author
Name - Aadra Khattri
