import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Loading the dataset
data = pd.read_csv('data/diabetes.csv')

# Target and Features
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model details
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluating
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Saving the model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
