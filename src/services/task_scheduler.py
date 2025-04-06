import os
import sys
import json
import datetime
from openai import OpenAI
from typing import List, Dict, Any
import logging

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database import SessionLocal
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.models import TrelloProjectTask
from src.schemas import TaskScheduleRequest, ScheduleResponse

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
SCHEDULE_HISTORY = os.path.join(history_dir, "tasks_schedules.json")

os.makedirs(history_dir, exist_ok=True)

# Initialize the history file if it doesn't exist.
if not os.path.exists(SCHEDULE_HISTORY):
    with open(SCHEDULE_HISTORY, "w") as f:
        json.dump([], f)

def get_trello_project_tasks() -> List[TrelloProjectTask]:
    session = SessionLocal()
    try:
        tasks = session.query(TrelloProjectTask).all()
        return tasks
    finally:
        session.close()

def sync_trello_tasks_to_json():
    project_tasks = get_trello_project_tasks()

    project_task_data = [
        {
            "project_id": task.project_id,
            "start_date": task.start_date.strftime('%Y-%m-%d'),  # Convert to string in YYYY-MM-DD format
            "end_date": task.event_date.strftime('%Y-%m-%d'),    # Convert to string in YYYY-MM-DD format
            "trello_board_data": json.loads(task.trello_board_data) if task.trello_board_data else None  # Parse JSON string to dictionary
        }
        for task in project_tasks
    ]

    with open(SCHEDULE_HISTORY, "w") as f:
        json.dump(project_task_data, f, indent=4)
    print(f"Synced {len(project_tasks)} project tasks to {SCHEDULE_HISTORY}")


def calculate_duration(start: str, end: str) -> int:
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    return (end_date - start_date).days

def clean_json_response(response_text: str) -> str:
    """
    Clean the response text by stripping markdown formatting.
    """
    return response_text.strip().strip("```json").strip("```").strip()

def save_schedule(schedule: dict):
    try:
        with open(SCHEDULE_HISTORY, "r+") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
            history.append(schedule)
            f.seek(0)
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving schedule: {e}")

def get_project_data_from_history(project_id: int) -> Dict[str, Any]:
    """
    Retrieve the project data from the SCHEDULE_HISTORY file based on the given project_id.
    """
    try:
        with open(SCHEDULE_HISTORY, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
    for record in data:
        if record.get("project_id") == project_id:
            return record
    return None

def create_schedule(request: TaskScheduleRequest) -> ScheduleResponse:
    # Retrieve project data from the SCHEDULE_HISTORY file based on project_id.
    project_data = get_project_data_from_history(request.project_id)
    if not project_data:
        raise ValueError(f"No project data found for project_id: {request.project_id}")

    # Prepare the AI prompt with project details.
    prompt = f"""
Project ID: {request.project_id}
Start Date: {project_data.get("start_date")}
End Date: {project_data.get("end_date")}
Trello Board Data: {json.dumps(project_data.get("trello_board_data", {}), indent=2)}
"""

    # Define the scheduling rules for the AI.
    rules = """
You are an AI assistant for generating project schedules. Follow these rules:

    - Generate a project schedule based on the provided categories and dates.
    - Ensure the schedule is clear, concise, and includes all necessary details.
    - Each category may have multiple tasks with specific due dates.
    - Due dates must be formatted as YYYY-MM-DD.
    - All due dates should fall within the project start and end dates.
    - Due dates must be unique and should not overlap within the same department.
    - Consider task dependencies based on task names (e.g., a task can only be scheduled after its dependent task is completed).

    EXPECTED OUTPUT:
    - A JSON object with the following structure:

    {
        "success": true,
        "project_id": <int>,  # The unique ID for the project
        "start": <string>,  # Project start date in YYYY-MM-DD format
        "end": <string>,  # Project end date in YYYY-MM-DD format
        "duration": <int>,  # Duration of the project in days
        "trello_tasks": {  # Mapping of departments to their respective tasks
            "<department_name>": {  # Name of the department
                "<category_name>": {  # Name of the category within the department
                    "<task_name_1>": "<due_date_1>",  # Task name and its due date in YYYY-MM-DD format
                    "<task_name_2>": "<due_date_2>",
                },
            }
        }
    }
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": rules},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=3000,
        top_p=1,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content
    cleaned_response = clean_json_response(response_text)
    try:
        schedule_dict = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse schedule JSON: {e}\nResponse Content: {cleaned_response}")

    if "start" in schedule_dict and "end" in schedule_dict:
        schedule_dict["duration"] = calculate_duration(schedule_dict["start"], schedule_dict["end"])

    schedule = ScheduleResponse(**schedule_dict)

    return schedule

if __name__ == "__main__":
    # Example: sync the project tasks before generating a schedule.
    # (Ensure that SCHEDULE_HISTORY contains the necessary project data.)
    sync_trello_tasks_to_json()

    dummy_request = TaskScheduleRequest(project_id=1)

    try:
        schedule = create_schedule(dummy_request)
        # save_schedule(schedule.model_dump())
        print(json.dumps(schedule.model_dump(), indent=4))
    except Exception as e:
        print(f"Error generating schedule: {e}")
