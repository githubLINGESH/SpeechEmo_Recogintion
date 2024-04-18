import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import time

def parse_features(feature_str):
    numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+", feature_str)
    numbers_float = [float(num) for num in numbers_str]
    fixed_length = 20800
    padded_array = np.zeros(fixed_length)
    padded_array[:len(numbers_float)] = numbers_float[:fixed_length]
    return padded_array

# Load dataset
print("Loading dataset...")
df = pd.read_csv('f_processed_features.csv')

# Prepare features and labels
print("Preparing features and labels...")
features = np.array([parse_features(x) for x in df['Features']])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['Label'])

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define SVM model with GridSearchCV
print("Defining SVM model with GridSearchCV...")
parameter_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

svm_model = SVC()
clf = GridSearchCV(svm_model, parameter_grid, cv=5, scoring='accuracy')

# Training
print("Training the model...")
start_time = time.time()
clf.fit(X_train_scaled, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Best model after grid search
print("Best model after grid search:")
best_model = clf.best_estimator_
print(best_model)

# Evaluate the model
print("Evaluating the model...")
y_pred = best_model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving the model, scaler, and label encoder
print("Saving the model, scaler, and label encoder...")
joblib.dump(best_model, 'emotion_svm_model.pkl')
joblib.dump(scaler, 'emotion_scaler_svm.pkl')
joblib.dump(label_encoder, 'emotion_label_encoder_svm.pkl')

print("Model, scaler, and label encoder saved successfully.")
