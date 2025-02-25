import random
import json
from datetime import datetime, timedelta

def generate_project_data(num_projects=5000):
    categories = [
        ("1 Year to 6 Months before", 180, 365),
        ("9 Months to 6 Months before", 270, 365),
        ("6 Months to 3 Months before", 90, 180),
        ("4 Months to 3 Months before", 120, 150),
        ("3 Months to 1 Month before", 30, 90),
        ("1 Month to 1 Week before", 7, 30),
        ("1 Week to 1 Day before", 1, 7),
        ("1 Week before and Wedding Day", 1, 7),
        ("Wedding Day", 0, 0),
        ("6 Months after Wedding Day", 180, 180)  # Ensuring 6 months after extends correctly
    ]
    
    projects = []
    for i in range(1, num_projects + 1):
        duration = random.randint(120, 650)
        start_date = datetime.strptime(f"202{random.randint(0, 5)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}", "%Y-%m-%d")
        end_date = start_date + timedelta(days=duration)
        
        task_categories = []
        for category, min_days, max_days in categories:
            if category == "6 Months after Wedding Day":
                start_day = duration  # Starts from wedding day
                end_day = duration + max_days  # Extends 180 days beyond wedding day
            else:
                start_day = max(0, duration - max_days)
                end_day = max(0, duration - min_days)
                if start_day > end_day:
                    continue  # Ensure valid range
            task_categories.append({
                "category": category,
                "date_range": [start_day, end_day]
            })
        
        projects.append({
            "project_name": f"project {i}",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "duration": duration,
            "task_category": task_categories
        })
    
    return projects

data = generate_project_data()
with open("src/datasets/task_scheduling_dataset.json", "w") as f:
    json.dump(data, f, indent=4)
