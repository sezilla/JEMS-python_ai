import json
import numpy as np
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATASET_PATH = 'src/datasets/task_classification_dataset.json'
BASE_MODEL_PATH = 'src/classifier/baseModel/task_classification_model.pkl'
NEW_MODEL_PATH = 'src/classifier/runningModels/task_classification_model.pkl'
HISTORY_PATH = 'src/classifier/history/classification_history.json'

THRESHOLD = 0.3  # Increased threshold for better confidence

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_dataset():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found!")
        return []
    with open(DATASET_PATH, 'r') as file:
        data = json.load(file)
        print(f"Loaded {len(data)} training examples")
        return data

def load_history():
    print("Loading classification history...")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as file:
            history = json.load(file)
            print(f"Loaded {len(history)} historical classifications")
            return history
    print("No history found, starting fresh")
    return []

def prepare_data(data):
    print("Preparing data for model training...")
    X = [item['task'] for item in data]
    y_labels, y_descriptions, departments = [], {}, set()
    
    for item in data:
        labels = [key for dep in item['department'] for key, value in dep.items()]
        for dep in item['department']:
            for key, value in dep.items():
                if key not in y_descriptions:
                    y_descriptions[key] = value
                departments.add(key)
        y_labels.append(labels)
    
    departments = sorted(departments)
    print(f"Found {len(departments)} departments: {', '.join(departments)}")
    
    y_multilabel = np.zeros((len(X), len(departments)), dtype=int)
    department_index = {dep: i for i, dep in enumerate(departments)}
    
    for i, labels in enumerate(y_labels):
        for label in labels:
            y_multilabel[i, department_index[label]] = 1
    
    return X, y_multilabel, y_descriptions, departments

def train_model(data, model_path):
    print(f"\nTraining new model with {len(data)} examples...")
    X, y_multilabel, descriptions, departments = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=7000, max_df=0.85, min_df=2, ngram_range=(1, 3))),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)))
    ])
    
    print("Training model...")
    model.fit(X_train, y_train)

    ensure_directory_exists(model_path)
    joblib.dump((model, descriptions, departments), model_path)
    print(f"Model saved to {model_path}")
    
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, target_names=departments))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    
    return model, descriptions, departments

def load_or_train_model():
    if os.path.exists(NEW_MODEL_PATH):
        print("Loading existing optimized model...")
        return joblib.load(NEW_MODEL_PATH)
    elif os.path.exists(BASE_MODEL_PATH):
        print("Loading base model and training a new optimized model...")
        data = load_dataset() + load_history()
        return train_model(data, NEW_MODEL_PATH)
    else:
        print("No existing model found. Training from scratch...")
        data = load_dataset()
        return train_model(data, BASE_MODEL_PATH)

def classify_task(task):
    print(f"\nClassifying task: {task}")
    model, descriptions, departments = load_or_train_model()
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', task) if len(s) > 5]
    print(f"Split into {len(sentences)} subtasks")
    
    classified_tasks = []
    
    for sentence in sentences:
        print(f"\nSubtask: {sentence}")
        
        # Get probabilities and predicted labels
        predicted_probs = model.predict_proba([sentence])
        predicted_labels = model.predict([sentence])[0]
        
        assigned_departments = []
        print("Department assignments (Department: Confidence Level):")
        
        for i, dep in enumerate(departments):
            # Extract probability correctly based on the classifier's output format
            probs = predicted_probs[i][0]
            prob = probs[1] if len(probs) > 1 else probs[0]
            
            print(f"  - {dep}: {prob:.4f}")
            
            if prob >= THRESHOLD:
                assigned_departments.append({dep: descriptions.get(dep, 'No description available')})
        
        if assigned_departments:
            classified_tasks.append({"task": sentence, "department": assigned_departments})
            print("Assigned departments:")
            for dept in assigned_departments:
                for dept_name, desc in dept.items():
                    print(f"  - {dept_name}: {desc}")
        else:
            print("No departments assigned (below confidence threshold)")
    
    # Save classification to history for future model improvement
    if classified_tasks:
        save_classification_history(classified_tasks)
    
    return classified_tasks

def save_classification_history(classifications):
    print("\nSaving classification history...")
    history = load_history()
    history.extend(classifications)
    ensure_directory_exists(HISTORY_PATH)
    with open(HISTORY_PATH, 'w') as file:
        json.dump(history, file, indent=4)
    print(f"Classification history updated with {len(classifications)} new entries.")

# Initialize model
if __name__ == "__main__":
    print("Initializing Task Classification Service...")
    model, descriptions, departments = load_or_train_model()
    
    # Example test
    example_task = "Create a beautifully decorated garden setting with fresh flowers, soft lighting, and elegant seating. Also, arrange a string quartet for the ceremony and hire a photographer to capture the event."
    print("\n" + "="*80)
    print("TESTING WITH EXAMPLE TASK")
    print("="*80)
    result = classify_task(example_task)
    print("\nFinal classification result:")
    print(json.dumps(result, indent=4))