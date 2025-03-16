import os
import sys
import json
import datetime
from openai import OpenAI
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
    TeamAllocation,
    Task
    )
from src.schemas import TeamAllocationRequest, TeamAllocationResponse

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

def get_package() -> List[Package]:
    session = SessionLocal()
    try:
        packages = session.query(Package).all()
        return packages
    finally:
        session.close()

def get_task() -> List[Task]:
    session = SessionLocal()
    try:
        tasks = session.query(Task).all()
        return tasks
    finally:
        session.close()

def get_task_package() -> List[TaskPackage]:
    session = SessionLocal()
    try:
        task_packages = session.query(TaskPackage).all()
        return task_packages
    finally:
        session.close()

def get_department_team() -> List[DepartmentTeam]:
    session = SessionLocal()
    try:
        department_teams = session.query(DepartmentTeam).all()
        return department_teams
    finally:
        session.close()

def get_team_allocation() -> List[TeamAllocation]:
    session = SessionLocal()
    try:
        allocated_teams = session.query(TeamAllocation).all()
        return allocated_teams
    finally:
        session.close()

def clean_json_response(response_text: str) -> Dict[str, Any]:
    """Cleans up AI-generated JSON response safely."""
    try:
        # Try to extract JSON if wrapped in code blocks
        response_text = response_text.strip()
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_content = response_text
            
        # Find the first occurrence of '{' and the last occurrence of '}'
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_content = json_content[start_idx:end_idx+1]
            
        result = json.loads(json_content)
        return result
    except Exception as e:
        # Return a default error structure if parsing fails
        return {
            "success": False,
            "error": f"Failed to parse response: {str(e)}",
            "raw_response": response_text
        }

def convert_to_safe_dict(obj: Any) -> Dict[str, Any]:
    """Convert SQLAlchemy objects to dictionaries with only their public attributes."""
    if obj is None:
        return {}
    
    result = {}
    # Get all non-private attributes
    for attr in dir(obj):
        if not attr.startswith('_') and not callable(getattr(obj, attr)):
            try:
                value = getattr(obj, attr)
                # Try to make the value JSON serializable
                if isinstance(value, datetime.date):
                    result[attr] = str(value)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[attr] = value
            except Exception:
                # Skip attributes that can't be accessed or serialized
                pass
    return result

def allocate_team(team_allocation: TeamAllocationRequest) -> TeamAllocationResponse:
    packages = get_package()
    tasks = get_task()
    task_packages = get_task_package()
    department_teams = get_department_team()
    history = get_team_allocation()

    # Convert objects to safe dictionaries
    package_data = [convert_to_safe_dict(p) for p in packages]
    task_data = [convert_to_safe_dict(t) for t in tasks]
    task_package_data = [convert_to_safe_dict(tp) for tp in task_packages]
    department_team_data = [convert_to_safe_dict(dt) for dt in department_teams]
    allocation_history = [convert_to_safe_dict(h) for h in history]

    prompt = f"""
You are a team allocation assistant. Based on the following data:

PROJECT INFO:
- Project ID: {team_allocation.project_id}
- Package ID: {team_allocation.package_id}
- Time span: {team_allocation.start} to {team_allocation.end}

ALLOCATION RULES:
- Only allocate teams from departments associated with package_id {team_allocation.package_id}
- Ensure teams are available during the requested timespan
- Balance workload across teams (use allocation_history to determine current loads)
- Teams can be allocated to multiple projects if necessary
- When all teams are busy, stack projects while maintaining balanced workload

DATA:
Packages: {json.dumps(package_data)}
Tasks: {json.dumps(task_data)}
Task-Package Relations: {json.dumps(task_package_data)}
Department Teams: {json.dumps(department_team_data)}
Current Allocations: {json.dumps(allocation_history)}

Return ONLY a JSON object in this format:
{{
    "success": true,
    "project_id": {team_allocation.project_id},
    "package_id": {team_allocation.package_id},
    "start": "{team_allocation.start}",
    "end": "{team_allocation.end}",
    "allocated_teams": [team_id1, team_id2, ...]
}}
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": team_allocation.model_dump_json()}
        ],
        max_tokens=3000,
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    try:
        output_text = response.choices[0].message.content
        result = clean_json_response(output_text)
        
        # Check if parsing failed
        if not result.get("success", False) and "error" in result:
            raise ValueError(f"AI returned invalid JSON: {result.get('error')}")
        
        # Convert the result to a TeamAllocationResponse
        return TeamAllocationResponse(
            success=result.get("success", False),
            project_id=result.get("project_id", team_allocation.project_id),
            package_id=result.get("package_id", team_allocation.package_id),
            start=result.get("start", team_allocation.start),
            end=result.get("end", team_allocation.end),
            allocated_teams=result.get("allocated_teams", [])
        )
    except Exception as e:
        raise ValueError(f"Failed to process allocation: {str(e)}")
    

if __name__ == "__main__":
    test_request = TeamAllocationRequest(
        project_id=1,
        package_id=2,
        start="2024-09-01",
        end="2025-07-05"
    )

    try:
        result = allocate_team(test_request)
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(json.dumps({"success": False, "message": "failed", "reason": str(e)}, indent=2))