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
    Task,
    TeamAllocation
    )
from src.schemas import TeamAllocationRequest, TeamAllocationResponse

client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

history_dir = "src/history"
ALLOCATION_HISTORY = os.path.join(history_dir, "team_allocation.json")

os.makedirs(history_dir, exist_ok=True)

if not os.path.exists(ALLOCATION_HISTORY):
    with open(ALLOCATION_HISTORY, "w") as f:
        json.dump([], f)

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

def get_db_allocation_history() -> List[TeamAllocation]:
    session = SessionLocal()
    try:
        allocations = session.query(TeamAllocation).all()
        return allocations
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
    """Cleans up AI-generated JSON response safely."""
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

def get_package_departments(package_id: int) -> List[int]:
    """Get departments associated with a specific package."""
    print(f"Getting departments for package ID: {package_id}")
    
    task_packages = get_task_package()
    tasks = get_task()
    
    task_package_data = [convert_to_safe_dict(tp) for tp in task_packages]
    task_data = [convert_to_safe_dict(t) for t in tasks]
    
    package_task_ids = [tp['task_id'] for tp in task_package_data 
                      if tp.get('package_id') == package_id]
    
    department_ids = [t['department_id'] for t in task_data 
                    if t.get('id') in package_task_ids]
    
    unique_departments = list(set(department_ids))
    print(f"Package {package_id} requires departments: {unique_departments}")
    return unique_departments

def get_department_teams(department_ids: List[int]) -> List[int]:
    """Get teams associated with given departments."""
    print(f"Getting teams for departments: {department_ids}")
    
    department_teams = get_department_team()
    
    department_team_data = [convert_to_safe_dict(dt) for dt in department_teams]
    
    team_ids = [dt['team_id'] for dt in department_team_data 
              if dt.get('department_id') in department_ids]
    
    unique_teams = list(set(team_ids))
    print(f"Found teams for departments {department_ids}: {unique_teams}")
    return unique_teams

def get_team_availability(team_ids: List[int], start_date: str, end_date: str) -> Dict[int, List[Dict]]:
    """Get availability of teams within specified date range."""
    print(f"Checking availability for teams {team_ids} from {start_date} to {end_date}")
    
    with open(ALLOCATION_HISTORY, "r") as f:
        allocation_data = json.load(f)
    
    team_schedules = {team_id: [] for team_id in team_ids}
    
    for allocation in allocation_data:
        if 'allocated_teams' not in allocation or not isinstance(allocation['allocated_teams'], list):
            continue
            
        for team_id in allocation['allocated_teams']:
            if team_id in team_ids:
                schedule_entry = {}
                if 'project_id' in allocation:
                    schedule_entry['project_id'] = allocation['project_id']
                if 'start' in allocation:
                    schedule_entry['start'] = allocation['start']
                if 'end' in allocation:
                    schedule_entry['end'] = allocation['end']
                    
                team_schedules[team_id].append(schedule_entry)
    
    return team_schedules

def check_date_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Check if two date ranges overlap."""
    start1_date = datetime.datetime.strptime(start1, "%Y-%m-%d").date()
    end1_date = datetime.datetime.strptime(end1, "%Y-%m-%d").date()
    start2_date = datetime.datetime.strptime(start2, "%Y-%m-%d").date()
    end2_date = datetime.datetime.strptime(end2, "%Y-%m-%d").date()
    
    return max(start1_date, start2_date) <= min(end1_date, end2_date)

def identify_available_teams(team_schedules: Dict[int, List[Dict]], start_date: str, end_date: str) -> Dict[int, bool]:
    """Identify which teams are available during the specified period."""
    available_teams = {}
    
    for team_id, schedules in team_schedules.items():
        is_available = True
        for schedule in schedules:
            if 'start' in schedule and 'end' in schedule:
                if check_date_overlap(start_date, end_date, schedule['start'], schedule['end']):
                    is_available = False
                    break
        available_teams[team_id] = is_available
    
    available_team_ids = [team_id for team_id, available in available_teams.items() if available]
    unavailable_team_ids = [team_id for team_id, available in available_teams.items() if not available]
    
    print(f"Available teams: {available_team_ids}")
    print(f"Unavailable teams (already allocated): {unavailable_team_ids}")
    
    return available_teams

def count_team_allocations(team_ids: List[int]) -> Dict[int, int]:
    """Count the number of existing allocations for each team."""
    with open(ALLOCATION_HISTORY, "r") as f:
        allocation_data = json.load(f)
    
    allocation_counts = {team_id: 0 for team_id in team_ids}
    
    for allocation in allocation_data:
        if 'allocated_teams' in allocation and isinstance(allocation['allocated_teams'], list):
            for team_id in allocation['allocated_teams']:
                if team_id in team_ids:
                    allocation_counts[team_id] += 1
    
    print(f"Team allocation counts: {allocation_counts}")
    return allocation_counts

def allocate_team(team_allocation: TeamAllocationRequest) -> TeamAllocationResponse:
    """Allocate teams to a project based on package requirements and team availability."""
    print(f"\nAllocating teams for Project ID: {team_allocation.project_id}, Package ID: {team_allocation.package_id}")
    print(f"Project timespan: {team_allocation.start} to {team_allocation.end}")
    
    packages = get_package()
    departments = get_department()
    package_data = [convert_to_safe_dict(p) for p in packages]
    department_data = [convert_to_safe_dict(d) for d in departments]
    
    package_dept_ids = get_package_departments(team_allocation.package_id)
    
    eligible_team_ids = get_department_teams(package_dept_ids)
    
    team_schedules = get_team_availability(eligible_team_ids, team_allocation.start, team_allocation.end)
    
    available_teams = identify_available_teams(team_schedules, team_allocation.start, team_allocation.end)
    
    allocation_counts = count_team_allocations(eligible_team_ids)
    
    dept_teams = get_department_team()
    dept_team_data = [convert_to_safe_dict(dt) for dt in dept_teams]
    team_to_dept = {}
    for dt in dept_team_data:
        if 'team_id' in dt and 'department_id' in dt:
            team_to_dept[dt['team_id']] = dt['department_id']
    
    dept_names = {d.get('id'): d.get('name', f"Department {d.get('id')}") for d in department_data if 'id' in d}

    dept_to_teams = {}
    for team_id, dept_id in team_to_dept.items():
        if dept_id in package_dept_ids and team_id in eligible_team_ids:
            if dept_id not in dept_to_teams:
                dept_to_teams[dept_id] = []
            dept_to_teams[dept_id].append(team_id)
    
    team_priority = {}
    for dept_id, teams in dept_to_teams.items():
        sorted_teams = sorted(teams, key=lambda team_id: (
            not available_teams.get(team_id, True),
            allocation_counts.get(team_id, 0)
        ))
        team_priority[dept_id] = sorted_teams

    prompt = f"""
You are a team allocation assistant. Based on the following data:

PROJECT INFO:
- Project ID: {team_allocation.project_id}
- Package ID: {team_allocation.package_id}
- Time span: {team_allocation.start} to {team_allocation.end}

ALLOCATION RULES:
- IMPORTANT: You must ONLY allocate teams from this pre-validated list of eligible teams: {eligible_team_ids}
- ENSURE ALL TEAMS GET ALLOCATED OVER TIME - balance workload across all teams
- Available teams (no timeline conflicts): {[team_id for team_id, available in available_teams.items() if available]}
- Teams with existing workload during timeline: {[team_id for team_id, available in available_teams.items() if not available]}
- Teams ordered by priority (least allocated first): {json.dumps(team_priority)}
- Team allocation counts: {json.dumps(allocation_counts)}
- Always prioritize teams with fewer existing allocations
- Teams with zero allocations should get highest priority
- Each required department must have at least one team allocated
- These are the only teams associated with departments that handle tasks for package
- When all teams are busy, stack projects while maintaining balanced workload
- Only allocate 1 team per department... meaning if a department has 2 teams available, you only allocate 1, if package requires 6 departments, you allocate only 1 team per department

DEPARTMENT INFORMATION:
- Package {team_allocation.package_id} requires departments: {package_dept_ids}
- Department names: {json.dumps(dept_names)}
- Team to Department mapping: {json.dumps(team_to_dept)}
- Department to Teams mapping: {json.dumps(dept_to_teams)}

DATA:
Packages: {json.dumps(package_data)}
Team Schedules: {json.dumps(team_schedules)}
Allocation Counts: {json.dumps(allocation_counts)}

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

IMPORTANT: 
1. Only include departments that are required for this package
2. Each team_id MUST be from the eligible_team_ids list
3. Prioritize teams with fewer existing allocations
4. Ensure every department has at least one team allocated
5. If multiple teams are available from the same department with similar allocation counts, prefer spreading the work across teams
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
        
        if not result.get("success", False) and "error" in result:
            raise ValueError(f"AI returned invalid JSON: {result.get('error')}")
        
        allocated_teams = []
        if "allocated_teams" in result and isinstance(result["allocated_teams"], list):
            for team_alloc in result["allocated_teams"]:
                if isinstance(team_alloc, dict) and "team_id" in team_alloc:
                    team_id = team_alloc["team_id"]
                    if team_id in eligible_team_ids:
                        allocated_teams.append(team_id)
        
        invalid_teams = [team for team in allocated_teams if team not in eligible_team_ids]
        if invalid_teams:
            print(f"WARNING: AI attempted to allocate invalid teams: {invalid_teams}")
            allocated_teams = [team for team in allocated_teams if team in eligible_team_ids]
        
        if not allocated_teams:
            print("WARNING: AI did not allocate any teams. Using fallback allocation.")
            for dept_id in package_dept_ids:
                if dept_id in dept_to_teams and dept_to_teams[dept_id]:
                    least_allocated_team = team_priority[dept_id][0]
                    allocated_teams.append(least_allocated_team)
        
        allocated_dept_ids = set()
        for team_id in allocated_teams:
            if team_id in team_to_dept:
                allocated_dept_ids.add(team_to_dept[team_id])
        
        for dept_id in package_dept_ids:
            if dept_id not in allocated_dept_ids and dept_id in dept_to_teams and dept_to_teams[dept_id]:
                least_allocated_team = team_priority[dept_id][0]
                allocated_teams.append(least_allocated_team)
                print(f"Added team {least_allocated_team} from department {dept_id} to ensure coverage")
        
        print(f"Final allocated teams: {allocated_teams}")
        
        # Return the final response
        return TeamAllocationResponse(
            success=True,
            project_id=team_allocation.project_id,
            package_id=team_allocation.package_id,
            start=team_allocation.start,
            end=team_allocation.end,
            allocated_teams=allocated_teams
        )
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise ValueError(f"Failed to process allocation: {str(e)}")
    
# def save_team_allocation(allocation: TeamAllocationResponse):
#     with open(ALLOCATION_HISTORY, "r") as f:
#         allocation_data = json.load(f)
    
#     allocation_data.append(allocation.model_dump())
    
#     with open(ALLOCATION_HISTORY, "w") as f:
#         json.dump(allocation_data, f, indent=4)

def sync_db_to_json():
    """Syncs database allocation history to the ALLOCATION_HISTORY JSON file."""
    allocations = get_db_allocation_history()

    # Convert to JSON serializable format
    allocation_data = [
        {
            "project_id": alloc.project_id,
            "package_id": alloc.package_id,
            "start_date": str(alloc.start_date),
            "end_date": str(alloc.end_date),
            "allocated_teams": alloc.allocated_teams or []
        }
        for alloc in allocations
    ]

    # Write to JSON file
    with open(ALLOCATION_HISTORY, "w") as f:
        json.dump(allocation_data, f, indent=4)

    print(f"Synced {len(allocations)} allocations to {ALLOCATION_HISTORY}")

if __name__ == "__main__":
    test_request = TeamAllocationRequest(
        project_id=1,
        package_id=1,
        start="2025-09-01",
        end="2026-07-05"
    )

    try:
        sync_db_to_json()
        result = allocate_team(test_request)
        # save_team_allocation(result)
        print("Generated Team Allocation:")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(json.dumps({"success": False, "message": "failed", "reason": str(e)}, indent=2))