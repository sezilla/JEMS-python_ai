#this is already Incremental learning... hirap nyan kasi incremental learning. pag wala model mag ttrain from scratch. pag meron model mag f-fine tune lang.
import json
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# Define paths
DATASET_PATH = os.path.join("datasets", "task_classification_dataset.json")
CLASSIFIER_FOLDER = "classifier"
MODEL_PATH = os.path.join(CLASSIFIER_FOLDER, "task_classifier.pkl")
LABEL_BINARIZER_PATH = os.path.join(CLASSIFIER_FOLDER, "label_binarizer.pkl")

# Load dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

tasks = [item["task"] for item in data]
departments = [item["departments"] for item in data]

# Load or initialize MultiLabelBinarizer
if os.path.exists(LABEL_BINARIZER_PATH):
    mlb = joblib.load(LABEL_BINARIZER_PATH)
    y = mlb.transform(departments)
else:
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(departments)

# Split new data for incremental training
X_train, X_test, y_train, y_test = train_test_split(tasks, y, test_size=0.2, random_state=42)

# Load existing model or create new one
if os.path.exists(MODEL_PATH):
    print("Loading existing model for fine-tuning...")
    pipeline = joblib.load(MODEL_PATH)
    
    # Transform the new training data using the existing TF-IDF vectorizer
    X_train_transformed = pipeline.named_steps["tfidf"].transform(X_train)
    
    # Fine-tune only the classifier (keep the vectorizer unchanged)
    pipeline.named_steps["clf"].fit(X_train_transformed, y_train)
else:
    print("No existing model found. Training from scratch...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),  # Convert text into TF-IDF features
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))  # Multi-label classification
    ])
    pipeline.fit(X_train, y_train)

# Save updated model
os.makedirs(CLASSIFIER_FOLDER, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
joblib.dump(mlb, LABEL_BINARIZER_PATH)

print(f"\nModel updated and saved in '{MODEL_PATH}'")
