import os
import sys
import json
from openai import OpenAI
import datetime
import logging
from typing import List, Dict, Any

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database import SessionLocal
from src.config import GITHUB_TOKEN, MODEL_NAME
from src.models import (
    Package,
    TaskPackage,
    DepartmentTeam,
    TeamUser,
    UserSkill,
    Team,
    Department,
    Task,
    Skill,
    User,
    )
from src.schemas import TeamAllocationRequest, TeamAllocationResponse

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

# base tables
def get_package() -> List[Package]:
    session = SessionLocal()
    try:
        package = session.query(Package).all()
        return package
    finally:
        session.close()
def get_skill() -> List[Skill]:
    session = SessionLocal()
    try:
        user_skill = session.query(Skill).all()
        return user_skill
    finally:
        session.close()

# relationship tables
def get_department_task() -> List[Task]:
    session = SessionLocal()
    try:
        department_task = session.query(Task).all()
        return department_task
    finally:
        session.close()
def get_task_package() -> List[TaskPackage]:
    session = SessionLocal()
    try:
        task_package = session.query(TaskPackage).all()
        return task_package
    finally:
        session.close()
def get_department_team() -> List[DepartmentTeam]:
    session = SessionLocal()
    try:
        department_team = session.query(DepartmentTeam).all()
        return department_team
    finally:
        session.close()
def get_team_user() -> List[TeamUser]:
    session = SessionLocal()
    try:
        team_user = session.query(TeamUser).all()
        return team_user
    finally:
        session.close()
def get_user_skill() -> List[UserSkill]:
    session = SessionLocal()
    try:
        user_skill = session.query(UserSkill).all()
        return user_skill
    finally:
        session.close()

def convert_to_safe_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    
    result = {}
    for attr in dir(obj):
        if not attr.startswith('_') and not callable(getattr(obj, attr)):
            try:
                value = getattr(obj, attr)
                if isinstance(value, datetime.date):
                    result[attr] = str(value)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[attr] = value
            except Exception:
                pass
    return result

def clean_json_response(response_text: str) -> Dict[str, Any]:
    try:
        response_text = response_text.strip()
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_content = response_text
            
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_content = json_content[start_idx:end_idx+1]
            
        result = json.loads(json_content)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse response: {str(e)}",
            "raw_response": response_text
        }
    
