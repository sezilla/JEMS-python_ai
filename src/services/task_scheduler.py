import os
import sys
import json
import datetime
from openai import OpenAI
import logging

# Suppress debug logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database import SessionLocal
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.models import Category

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key = GITHUB_TOKEN
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
history_file = os.path.join(history_dir, "category_schedules.json")

# Ensure history directory exists
os.makedirs(history_dir, exist_ok=True)

# Ensure history file exists
if not os.path.exists(history_file):
    with open(history_file, "w") as f:
        json.dump([], f)

def get_categories():
    session = SessionLocal()
    try:
        categories = session.query(Category).all()
        return categories
    finally:
        session.close()

def calculate_duration(start: str, end: str) -> int:
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    return (end_date - start_date).days

def clean_json_response(response_text: str) -> str:
    """Removes markdown JSON formatting from response."""
    return response_text.strip().strip("```json").strip("```")

def generate_schedule(project_id: int, start: str, end: str) -> dict:
    duration = calculate_duration(start, end)
    categories = get_categories()
    category_names = [cat.name for cat in categories]

    prompt = f"""
You are an AI scheduler for task categories. The scheduling rules:
  - Assign schedules based on project span.
  - Adjust proportionally to the project's duration.
  - Ensure minimal slots for short projects.
  - Evenly distribute if possible, with the last category absorbing extra days.

Project Details:
  Project ID: {project_id}
  Start Date: {start}
  End Date: {end}
  Project Duration: {duration} days

Available Categories:
  {', '.join(category_names)}

Please generate a schedule for each category. For each category, assign a start date and a deadline date within the project timeline. Output your response as a JSON object with the following structure:
{{
  "project_id": <int>,
  "start": "<YYYY-MM-DD>",
  "end": "<YYYY-MM-DD>",
  "duration": <int>,
  "categories": [
      ["Category Name", "<Start Date>", "<Deadline Date>"]
  ]
}}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    try:
        output_text = response.choices[0].message.content
        cleaned_json = clean_json_response(output_text)
        schedule = json.loads(cleaned_json)
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        raise ValueError(f"Failed to parse JSON from GPT response: {output_text}") from e

    return schedule

def save_schedule(schedule: dict):
    try:
        with open(history_file, "r+") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            history.append(schedule)
            f.seek(0)
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving schedule: {e}")

if __name__ == "__main__":
    test_project_id = 1
    test_start = "2023-01-01"
    test_end = "2023-09-01"
    
    try:
        schedule = generate_schedule(test_project_id, test_start, test_end)
        save_schedule(schedule)
        print("Generated Schedule:")
        print(json.dumps(schedule, indent=4))
    except Exception as e:
        print(f"Error generating schedule: {e}")