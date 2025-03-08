import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATASET_PATH = 'src/datasets/task_classification_dataset.json'
MODEL_PATH = 'src/classifier/baseModels/task_classification_model.pkl'
THRESHOLD = 0.2  # Lowered the minimum probability threshold to 20%

# Load dataset
def load_dataset():
    with open(DATASET_PATH, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(data):
    X = [item['task'] for item in data]
    y_labels = []
    y_descriptions = {}
    departments = set()
    
    for item in data:
        labels = []
        for dep in item['department']:
            for key, value in dep.items():
                labels.append(key)
                if key not in y_descriptions:
                    y_descriptions[key] = value  # Store one description per department
                departments.add(key)
        y_labels.append(labels)
    
    departments = sorted(departments)
    y_multilabel = np.zeros((len(X), len(departments)), dtype=int)
    
    department_index = {dep: i for i, dep in enumerate(departments)}
    for i, labels in enumerate(y_labels):
        for label in labels:
            y_multilabel[i, department_index[label]] = 1
    
    return X, y_multilabel, y_descriptions, departments

def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = joblib.load(MODEL_PATH)
    else:
        print("Training new model...")
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=1, ngram_range=(1,2))),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
    return model

# Load dataset and prepare data
data = load_dataset()
X, y_multilabel, y_descriptions, departments = prepare_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.2, random_state=42)

# Load or train model
pipeline = load_or_train_model(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=departments))
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")

def split_sentences(text):
    return [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if len(sentence) > 5]  # Ignore very short sentences

def classify_task(task):
    print(f"Classifying task: {task}")
    sentences = split_sentences(task)
    all_assigned_tasks = {}
    
    for sentence in sentences:
        predicted_probs = pipeline.predict_proba([sentence])
        predicted_labels = pipeline.predict([sentence])[0]
        
        for i, dep in enumerate(departments):
            if isinstance(predicted_probs[i], np.ndarray) and len(predicted_probs[i]) > 0:
                prob = predicted_probs[i][0][1] if len(predicted_probs[i][0]) > 1 else predicted_probs[i][0][0]
            else:
                prob = 1.0  # Default to high confidence if probs are not returned properly
            
            if prob >= THRESHOLD:
                if dep not in all_assigned_tasks:
                    all_assigned_tasks[dep] = {
                        "task": y_descriptions.get(dep, 'Task description not available'),
                        "confidence": prob * 100
                    }
                else:
                    # Merge confidence scores if multiple sentences contribute to the same department
                    all_assigned_tasks[dep]["confidence"] = max(all_assigned_tasks[dep]["confidence"], prob * 100)
    
    if not all_assigned_tasks:
        print("No confident classification detected. Consider rephrasing the task.")
    else:
        for dep in all_assigned_tasks:
            all_assigned_tasks[dep]["confidence"] = f"{all_assigned_tasks[dep]['confidence']:.2f}%"
        print(f"Assigned Departments and Confidence Scores: {all_assigned_tasks}")
    
    return all_assigned_tasks

# Example usage
# new_task = "Create a beautifully decorated garden setting with fresh flowers, soft lighting, and elegant seating for an intimate and romantic wedding ceremony. Serve a carefully curated multi-course meal featuring the couple’s favorite cuisines, paired with signature cocktails and a decadent dessert station. Arrange a live band for the ceremony and dinner, followed by a DJ playing an energetic mix of music to keep guests dancing all night. Capture every special moment with high-quality photography and videography, including a pre-wedding shoot and candid guest interactions. Provide professional hair and makeup services for the couple and wedding party, ensuring everyone looks flawless for the big day."
# print("Classified Task Output:")
# result = classify_task(new_task)
# print(result)
