import os
import sys
import json
from datetime import datetime, timedelta
from openai import OpenAI
from sqlalchemy.orm import Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import Category, Project
from src.schemas import CategoryScheduleRequest
from src.database import SessionLocal  # Use existing database connection

def get_openai_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is not set in environment variables")
    
    return OpenAI(api_key=token, base_url="https://models.inference.ai.azure.com")

def get_categories(db: Session):
    return [category.name for category in db.query(Category).all()]

def calculate_schedule(start_date: str, end_date: str, db: Session):
    start, end = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
    duration = (end - start).days

    categories = [category.name for category in db.query(Category).all()]
    category_count = len(categories)
    if not category_count:
        return []
    
    # Assign at least one day per category if project duration is too short
    if duration < category_count:
        return [
            {"category": cat, "start": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "end": (start + timedelta(days=i)).strftime("%Y-%m-%d")}
            for i, cat in enumerate(categories) if start + timedelta(days=i) <= end
        ]

    # Evenly distribute time across categories
    segment_size = duration // category_count
    category_schedule = []
    current_start = start

    for i, category in enumerate(categories):
        current_end = end if i == category_count - 1 else current_start + timedelta(days=segment_size)
        category_schedule.append({
            "category": category,
            "start": current_start.strftime("%Y-%m-%d"),
            "end": current_end.strftime("%Y-%m-%d")
        })
        current_start = current_end + timedelta(days=1)
        if current_start > end:
            break
    
    return category_schedule

def generate_schedule(request: CategoryScheduleRequest, db: Session):
    return {
        "project_id": request.project_id,
        "start": request.start,
        "end": request.end,
        "duration": (datetime.strptime(request.end, "%Y-%m-%d") - datetime.strptime(request.start, "%Y-%m-%d")).days,
        "category_schedule": calculate_schedule(request.start, request.end, db)
    }

def main():
    db = SessionLocal()
    
    # Ensure a test project exists
    if not db.query(Project).filter_by(id=1).first():
        test_project = Project(id=1, name="Test Project")
        db.add(test_project)
        db.commit()
        
    request = CategoryScheduleRequest(project_id=1, start="2025-06-01", end="2025-12-01")
    schedule = generate_schedule(request, db)
    
    client = get_openai_client()
    system_prompt = f"""
    You are an AI scheduler for task categories. The scheduling rules:
    - Assign schedules based on project span.
    - Adjust proportionally to the project’s duration.
    - Ensure minimal slots for short projects.
    - Evenly distribute if possible, with the last category absorbing extra days.
    Given the project from {request.start} to {request.end}:
    {json.dumps(schedule['category_schedule'], indent=4)}
    Optimize this schedule following the rules.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1
    )
    
    print(json.dumps(schedule, indent=4))

if __name__ == "__main__":
    main()
