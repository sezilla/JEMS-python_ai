import json
import numpy as np
import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATASET_PATH = 'src/datasets/task_classification_dataset.json'
MODEL_PATH = 'src/classifier/baseModel/task_classification_model.pkl'
THRESHOLD = 0.3

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Ensured directory exists: {os.path.dirname(path)}")

def load_dataset():
    print("\nLoading dataset...")
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found!")
        return []
    with open(DATASET_PATH, 'r') as file:
        data = json.load(file)
        print(f"Successfully loaded {len(data)} training examples")
        return data

def prepare_data(data):
    print("\nPreparing data for model training...")
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
    print(f"Found {len(departments)} departments:")
    for dept in departments:
        print(f"  - {dept}: {y_descriptions.get(dept, 'No description')}")
    
    y_multilabel = np.zeros((len(X), len(departments)), dtype=int)
    department_index = {dep: i for i, dep in enumerate(departments)}
    
    for i, labels in enumerate(y_labels):
        for label in labels:
            y_multilabel[i, department_index[label]] = 1
    
    print(f"Prepared {len(X)} tasks with {len(departments)} department labels")
    return X, y_multilabel, y_descriptions, departments

def train_model():
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING")
    print("="*80)
    
    data = load_dataset()
    if not data:
        print("No data available for training. Exiting.")
        return None
    
    X, y_multilabel, descriptions, departments = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multilabel, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Create and train the model
    print("\nInitializing model pipeline...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=7000, max_df=0.85, min_df=2, ngram_range=(1, 3))),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)))
    ])
    
    print("Training model - this may take a few minutes...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\nModel Performance Metrics:")
    print(classification_report(y_test, y_pred, target_names=departments))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Save the model
    print("\nSaving trained model...")
    ensure_directory_exists(MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        print(f"Overwriting existing model at {MODEL_PATH}")
    joblib.dump((model, descriptions, departments), MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
    
    return model, descriptions, departments

def test_model(model, descriptions, departments):
    print("\n" + "="*80)
    print("TESTING MODEL WITH EXAMPLE TASKS")
    print("="*80)
    
    # Complex example with multiple departments and tasks
    example_tasks = [
        "Create a beautifully decorated garden setting with fresh flowers, soft lighting, and elegant seating. Also, arrange a string quartet for the ceremony and hire a photographer to capture the event.",
        "Design custom wedding invitations with calligraphy and organize a catering service for 200 guests with dietary restrictions.",
        "Coordinate vendor arrivals and setup timelines for the day of the wedding."
    ]
    
    for task in example_tasks:
        print(f"\nTesting task: {task}")
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', task) if len(s) > 5]
        for sentence in sentences:
            print(f"\nSubtask: {sentence}")
            
            # Get probabilities and predicted labels
            predicted_probs = model.predict_proba([sentence])
            predicted_labels = model.predict([sentence])[0]
            
            print("Department confidence levels:")
            assigned_departments = []
            
            for i, dep in enumerate(departments):
                # Extract probability correctly
                probs = predicted_probs[i][0]
                prob = probs[1] if len(probs) > 1 else probs[0]
                
                confidence_str = f"{prob:.4f}" + (" ✓" if prob >= THRESHOLD else "")
                print(f"  - {dep}: {confidence_str}")
                
                if prob >= THRESHOLD:
                    assigned_departments.append({dep: descriptions.get(dep, 'No description')})
            
            if assigned_departments:
                print("\nAssigned departments and tasks:")
                for dept in assigned_departments:
                    for dept_name, desc in dept.items():
                        print(f"  - {dept_name}: {desc}")
            else:
                print("\nNo departments met the confidence threshold")

if __name__ == "__main__":
    print("Task Classifier Trainer")
    print("="*80)
    
    try:
        model, descriptions, departments = train_model()
        if model:
            test_model(model, descriptions, departments)
            print("\nTrainer completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")