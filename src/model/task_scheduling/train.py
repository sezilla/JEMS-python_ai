import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
with open("src/datasets/task_scheduling_dataset.json", "r") as f:
    data = json.load(f)
print(f"Dataset loaded successfully with {len(data)} projects.")

# Extract features and labels
print("Processing dataset...")
X = []
y = []
category_labels = set()

for project in data:
    start_date = datetime.strptime(project["start"], "%Y-%m-%d")
    end_date = datetime.strptime(project["end"], "%Y-%m-%d")
    duration = (end_date - start_date).days

    for task in project["task_category"]:
        category = task["category"]
        start_day, end_day = task["date_range"]
        X.append([duration, start_day, end_day])
        y.append(category)
        category_labels.add(category)

print(f"Extracted {len(X)} samples with {len(category_labels)} unique categories.")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Labels encoded.")

# Split dataset
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and encoder
print("Saving model and label encoder...")
joblib.dump(model, "src/classifier/task_scheduling_classifier.pkl")
joblib.dump(label_encoder, "src/classifier/label_encoder.pkl")
print("Model and encoder saved.")
