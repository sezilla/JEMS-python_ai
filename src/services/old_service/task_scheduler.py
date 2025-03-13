import json
import joblib
import numpy as np
from datetime import datetime
from pydantic import BaseModel

# Load model and label encoder
model = joblib.load("src/classifier/task_scheduling_classifier.pkl")
label_encoder = joblib.load("src/classifier/label_encoder.pkl")

# Define categories in correct order
categories = [
    ("1 Year to 6 Months before", 180, 365),
    ("9 Months to 6 Months before", 270, 365),
    ("6 Months to 3 Months before", 90, 180),
    ("4 Months to 3 Months before", 120, 150),
    ("3 Months to 1 Month before", 30, 90),
    ("1 Month to 1 Week before", 7, 30),
    ("1 Week before and Wedding Day", 1, 7),
    ("Wedding Day", 0, 0),
    ("6 Months after Wedding Day", 180, 180)
]

def predict_categories(project_name, start, end):
    """Predicts categories for a project based on start and end dates."""
    print(f"\nPredicting categories for project: {project_name}")

    # Convert dates to datetime
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    duration = (end_date - start_date).days

    predicted_categories = []
    seen_categories = set()  # Avoid duplicates

    for category, min_days, max_days in categories:
        if category == "6 Months after Wedding Day":
            start_day = duration  # Starts from wedding day
            end_day = duration + max_days  # Extends 180 days beyond wedding day
        else:
            start_day = max(0, duration - max_days)
            end_day = max(0, duration - min_days)

        if start_day > end_day:
            continue  # Skip invalid categories

        # Predict category
        input_data = np.array([[duration, start_day, end_day]])
        predicted_label = model.predict(input_data)
        predicted_category = label_encoder.inverse_transform(predicted_label)[0]

        if predicted_category not in seen_categories:
            predicted_categories.append({"category": predicted_category, "date_range": [start_day, end_day]})
            seen_categories.add(predicted_category)

    # Print predicted categories
    print("\nPredicted Categories with Date Ranges:")
    for cat in predicted_categories:
        print(f"Category: {cat['category']} | Date Range: {cat['date_range']}")

    return predicted_categories

class PredictRequest(BaseModel):
    project_name: str
    start: str
    end: str

# Example test
test_project_name = "Test Project"
test_start = "2023-01-01"
test_end = "2023-12-31"  # A full-year project
predicted_results = predict_categories(test_project_name, test_start, test_end)
