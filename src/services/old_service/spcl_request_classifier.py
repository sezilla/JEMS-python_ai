import json
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_PATH = 'src/datasets/task_classification_dataset.json'

BASE_MODEL_PATH = 'src/classifier/baseModel/task_classification_model.pkl'
NEW_MODEL_PATH = 'src/classifier/runningModels/task_classification_model.pkl'

HISTORY_PATH = 'src/classifier/history/classification_history.json'
THRESHOLD = 0.2

def load_dataset():
    with open(DATASET_PATH, 'r') as file:
        data = json.load(file)
    return data

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as file:
            return json.load(file)
    return []

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
                    y_descriptions[key] = value
                departments.add(key)
        y_labels.append(labels)
    
    departments = sorted(departments)
    y_multilabel = np.zeros((len(X), len(departments)), dtype=int)
    department_index = {dep: i for i, dep in enumerate(departments)}
    
    for i, labels in enumerate(y_labels):
        for label in labels:
            y_multilabel[i, department_index[label]] = 1
    
    return X, y_multilabel, y_descriptions, departments

def update_model_with_history():
    print("Updating model with history...")
    history = load_history()
    if not history:
        print("No history found, skipping update.")
        return
    
    updated_data = load_dataset() + history
    X, y_multilabel, _, _ = prepare_data(updated_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=1, ngram_range=(1,2))),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, BASE_MODEL_PATH)
    print("Model updated with classification history.")

def load_or_train_model():
    if os.path.exists(BASE_MODEL_PATH):
        print("Loading existing model...")
        return joblib.load(BASE_MODEL_PATH)
    
    print("Training new model...")
    data = load_dataset()
    X, y_multilabel, _, _ = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, max_df=0.9, min_df=1, ngram_range=(1,2))),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, BASE_MODEL_PATH)
    return model

def split_sentences(text):
    return [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if len(sentence) > 5]

def classify_task(task):
    print(f"Classifying task: {task}")
    update_model_with_history()
    model = load_or_train_model()
    
    sentences = split_sentences(task)
    all_assigned_tasks = []
    
    for sentence in sentences:
        predicted_probs = model.predict_proba([sentence])
        predicted_labels = model.predict([sentence])[0]
        
        assigned_departments = []
        for i, dep in enumerate(departments):
            prob = predicted_probs[i][0][1] if len(predicted_probs[i][0]) > 1 else predicted_probs[i][0][0]
            if prob >= THRESHOLD:
                assigned_departments.append({dep: y_descriptions.get(dep, 'Task description not available')})
        
        if assigned_departments:
            all_assigned_tasks.append({"task": sentence, "department": assigned_departments})
    
    save_classification_history(all_assigned_tasks)
    return all_assigned_tasks

def save_classification_history(classifications):
    history = load_history()
    history.extend(classifications)
    
    with open(HISTORY_PATH, 'w') as file:
        json.dump(history, file, indent=4)
    print("Classification saved to history.")

# Load dataset and train model initially
data = load_dataset()
X, y_multilabel, y_descriptions, departments = prepare_data(data)

# Load or train model
pipeline = load_or_train_model()

# Evaluate model
y_pred = pipeline.predict(X)
accuracy = accuracy_score(y_multilabel, y_pred)
print("Model Evaluation:")
print(classification_report(y_multilabel, y_pred, target_names=departments))
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
