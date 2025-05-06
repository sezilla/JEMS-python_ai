import os
import re
import sys
import json
import traceback
import logging
from typing import List, Dict, Any
from openai import OpenAI
from src.database import SessionLocal
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.schemas import CardData, CheckItem, Checklist, TaskAllocationRequest, TaskAllocationResponse, UserData

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
TASK_ALLOCATION_HISTORY = os.path.join(history_dir, "task_allocation.json")

os.makedirs(history_dir, exist_ok=True)

if not os.path.exists(TASK_ALLOCATION_HISTORY):
    with open(TASK_ALLOCATION_HISTORY, "w") as f:
        json.dump([], f)

def clean_json_response(response_text: str) -> str:
    start = response_text.find('{')
    end = response_text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found in response.")
    cleaned = response_text[start:end+1]
    cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
    return cleaned

def is_json_well_formed(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def allocate_tasks(request: TaskAllocationRequest) -> TaskAllocationResponse:
    try:
        project_id = request.project_id
        data_array = request.data_array
        users_by_department = request.users

        task_summary = []
        for card in data_array:
            for checklist in card.checklists:
                for item in checklist.check_items:
                    task_summary.append({
                        "card_id": card.card_id,
                        "checklist_id": checklist.checklist_id,
                        "checklist_name": checklist.checklist_name,
                        "check_item_id": item.check_item_id,
                        "check_item_name": item.check_item_name
                    })

        rules = """
        You are an AI that allocates tasks to users based on skills and department.
        - Each card represents a department.
        - Only users from that department should be assigned to that card's checklists.
        - Match users to tasks (check_items) based on skills.
        - Spread tasks fairly.
        - IMPORTANT!!! The final JSON response must use checklist_id as the key.
        - Final JSON format should be:
          {
            "checklist_id_1": [
              {"user_id": int, "card_id": str, "check_item_id": str, "check_item_name": str},
              ...
            ],
            "checklist_id_2": [
              ...
            ]
          }
        """

        users_dict = {
            dept: [user.model_dump() for user in user_list]
            for dept, user_list in users_by_department.items()
        }

        prompt = f"""
        Project ID: {project_id}
        Tasks:
        {json.dumps(task_summary, indent=2)}

        Users by Department:
        {json.dumps(users_dict, indent=2)}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": rules},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.7
        )

        response_text = response.choices[0].message.content
        print(f"Raw model response:\n{response_text}")

        cleaned_response = clean_json_response(response_text)

        if not is_json_well_formed(cleaned_response):
            raise ValueError("Model response JSON is not well-formed.")

        parsed = json.loads(cleaned_response)

        result = TaskAllocationResponse(
            success=True,
            project_id=request.project_id,
            checklists=parsed
        )

        print(f"Response object content: {result.model_dump()}")
        return result

    except Exception as e:
        logging.error(f"Error in task allocation: {e}")
        traceback.print_exc()
        return TaskAllocationResponse(
            success=False,
            project_id=request.project_id,
            checklists={},
            error=str(e)
        )

if __name__ == "__main__":
    from pydantic import BaseModel
    import uuid

    def uid():
        return str(uuid.uuid4())[:24]

    dummy_request = TaskAllocationRequest(
        project_id=1,
        data_array=[
            CardData(
                card_id=uid(),
                card_name="Design",
                checklists=[
                    Checklist(
                        checklist_id="67ff892b923c801dfa8cf7e8",
                        checklist_name="Design Checklist",
                        check_items=[
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cf9ff",
                                check_item_name="Create mockups"
                            ),
                        ]
                    )
                ]
            ),
            CardData(
                card_id=uid(),
                card_name="Development",
                checklists=[
                    Checklist(
                        checklist_id="67ff892b923c801dfa8cf7eb",
                        checklist_name="Development Checklist",
                        check_items=[
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cfa0c",
                                check_item_name="Build login API"
                            ),
                            CheckItem(
                                check_item_id="67ff892c923c801dfa8cfa0d",
                                check_item_name="Setup database schema"
                            )
                        ]
                    )
                ]
            )
        ],
        users={
            "Design": [
                UserData(user_id=188, skills=["Figma", "UX", "Wireframes"]),
                UserData(user_id=194, skills=["Branding", "Illustrator"])
            ],
            "Development": [
                UserData(user_id=250, skills=["Laravel", "MySQL"]),
                UserData(user_id=194, skills=["API", "PHP", "Databases"])
            ]
        }
    )

    try:
        schedule = allocate_tasks(dummy_request)
        print("\n=== Allocation Result ===")
        print(json.dumps(schedule.model_dump(), indent=4))
    except Exception as e:
        print(f"Error generating schedule: {e}")
