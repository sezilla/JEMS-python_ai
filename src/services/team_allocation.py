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
    Department,
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

def get_department() -> List[Department]:
    session = SessionLocal()
    try:
        departments = session.query(Department).all()
        return departments
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

def get_package_departments(package_id: int) -> List[int]:
    """Get departments associated with a specific package."""
    # Get all tasks associated with the package
    task_packages = get_task_package()
    tasks = get_task()
    
    # Convert to dictionaries for easier processing
    task_package_data = [convert_to_safe_dict(tp) for tp in task_packages]
    task_data = [convert_to_safe_dict(t) for t in tasks]
    
    # Find tasks associated with the package
    package_task_ids = [tp['task_id'] for tp in task_package_data 
                      if tp.get('package_id') == package_id]
    
    # Find departments associated with those tasks
    department_ids = [t['department_id'] for t in task_data 
                    if t.get('id') in package_task_ids]
    
    return list(set(department_ids))  # Remove duplicates

def get_department_teams(department_ids: List[int]) -> List[int]:
    """Get teams associated with given departments."""
    department_teams = get_department_team()
    
    # Convert to dictionaries for easier processing
    department_team_data = [convert_to_safe_dict(dt) for dt in department_teams]
    
    # Find teams belonging to the specified departments
    team_ids = [dt['team_id'] for dt in department_team_data 
              if dt.get('department_id') in department_ids]
    
    return list(set(team_ids))  # Remove duplicates

def get_team_availability(team_ids: List[int], start_date: str, end_date: str) -> Dict[int, List[Dict]]:
    """Get availability of teams within specified date range."""
    allocations = get_team_allocation()
    
    # Convert to dictionaries for easier processing
    allocation_data = [convert_to_safe_dict(a) for a in allocations]
    
    # Track allocations for each team
    team_schedules = {team_id: [] for team_id in team_ids}
    
    for allocation in allocation_data:
        # Check if allocated_teams exists and is a valid list
        if 'allocated_teams' not in allocation or not isinstance(allocation['allocated_teams'], list):
            continue
            
        for team_id in allocation['allocated_teams']:
            if team_id in team_ids:
                # Only add fields that we know exist
                schedule_entry = {}
                if 'project_id' in allocation:
                    schedule_entry['project_id'] = allocation['project_id']
                if 'start' in allocation:
                    schedule_entry['start'] = allocation['start']
                if 'end' in allocation:
                    schedule_entry['end'] = allocation['end']
                    
                team_schedules[team_id].append(schedule_entry)
    
    return team_schedules

def allocate_team(team_allocation: TeamAllocationRequest) -> TeamAllocationResponse:
    """Allocate teams to a project based on package requirements and team availability."""
    # Get all packages, departments, and tasks data
    packages = get_package()
    departments = get_department()
    package_data = [convert_to_safe_dict(p) for p in packages]
    department_data = [convert_to_safe_dict(d) for d in departments]
    
    # Get departments required for the package
    package_dept_ids = get_package_departments(team_allocation.package_id)
    
    # Get teams associated with those departments
    eligible_team_ids = get_department_teams(package_dept_ids)
    
    # Get team availability
    team_schedules = get_team_availability(eligible_team_ids, team_allocation.start, team_allocation.end)
    
    # Get current allocations for workload balancing
    allocation_history = [convert_to_safe_dict(h) for h in get_team_allocation()]
    
    # Log the eligible relationships for debugging
    print(f"Package {team_allocation.package_id} is associated with:")
    print(f"- Department IDs: {package_dept_ids}")
    print(f"- Eligible Team IDs: {eligible_team_ids}")
    
    # Create a mapping of teams to departments
    dept_teams = get_department_team()
    dept_team_data = [convert_to_safe_dict(dt) for dt in dept_teams]
    team_to_dept = {}
    for dt in dept_team_data:
        if 'team_id' in dt and 'department_id' in dt:
            team_to_dept[dt['team_id']] = dt['department_id']
    
    # Create a mapping of department IDs to names
    dept_names = {d.get('id'): d.get('name', f"Department {d.get('id')}") for d in department_data if 'id' in d}

    prompt = f"""
You are a team allocation assistant. Based on the following data:

PROJECT INFO:
- Project ID: {team_allocation.project_id}
- Package ID: {team_allocation.package_id}
- Time span: {team_allocation.start} to {team_allocation.end}

ALLOCATION RULES:
- IMPORTANT: You must ONLY allocate teams from this pre-validated list of eligible teams: {eligible_team_ids}
- These are the only teams associated with departments that handle tasks for package {team_allocation.package_id}
- Ensure teams are available during the requested timespan
- Balance workload across teams (use allocation_history to determine current loads)
- Teams can be allocated to multiple projects if necessary
- When all teams are busy, stack projects while maintaining balanced workload

DEPARTMENT INFORMATION:
- Package {team_allocation.package_id} requires departments: {package_dept_ids}
- Department names: {json.dumps(dept_names)}
- Team to Department mapping: {json.dumps(team_to_dept)}

DATA:
Packages: {json.dumps(package_data)}
Team Schedules: {json.dumps(team_schedules)}
Current Allocations: {json.dumps(allocation_history)}

Return ONLY a JSON object in this format:
{{
    "success": true,
    "project_id": {team_allocation.project_id},
    "package_id": {team_allocation.package_id},
    "start": "{team_allocation.start}",
    "end": "{team_allocation.end}",
    "allocated_teams": [
        {{"department_name": "department_name1", "team_id": team_id1}},
        {{"department_name": "department_name2", "team_id": team_id2}},
        ...
    ]
}}

IMPORTANT: Only include departments that are required for this package. Each team_id MUST be from the eligible_team_ids list.
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": team_allocation.model_dump_json()}
        ],
        max_tokens=2000,
        temperature=1,
        response_format={"type": "json_object"}
    )

    try:
        output_text = response.choices[0].message.content
        result = clean_json_response(output_text)
        
        # Check if parsing failed
        if not result.get("success", False) and "error" in result:
            raise ValueError(f"AI returned invalid JSON: {result.get('error')}")
        
        # Extract just the team IDs for the response
        allocated_teams = []
        if "allocated_teams" in result and isinstance(result["allocated_teams"], list):
            for team_alloc in result["allocated_teams"]:
                if isinstance(team_alloc, dict) and "team_id" in team_alloc:
                    team_id = team_alloc["team_id"]
                    if team_id in eligible_team_ids:
                        allocated_teams.append(team_id)
        
        # Validate that we only have eligible teams
        invalid_teams = [team for team in allocated_teams if team not in eligible_team_ids]
        if invalid_teams:
            print(f"WARNING: AI attempted to allocate invalid teams: {invalid_teams}")
            # Filter out invalid teams
            allocated_teams = [team for team in allocated_teams if team in eligible_team_ids]
        
        # Return the final response
        return TeamAllocationResponse(
            success=result.get("success", True),  # Default to True if not specified
            project_id=result.get("project_id", team_allocation.project_id),
            package_id=result.get("package_id", team_allocation.package_id),
            start=result.get("start", team_allocation.start),
            end=result.get("end", team_allocation.end),
            allocated_teams=allocated_teams
        )
    except Exception as e:
        raise ValueError(f"Failed to process allocation: {str(e)}")

if __name__ == "__main__":
    test_request = TeamAllocationRequest(
        project_id=1,
        package_id=1,
        start="2025-03-19",
        end="2026-03-19"
    )

    try:
        result = allocate_team(test_request)
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(json.dumps({"success": False, "message": "failed", "reason": str(e)}, indent=2))