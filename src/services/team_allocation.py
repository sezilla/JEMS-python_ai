import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import select

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now imports from src should work
from src.models import TaskPackage, Task, ProjectTeam, DepartmentTeam
from src.database import get_db

# Initialize OpenAI client
def get_openai_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN is not set in environment variables")

    endpoint = "https://models.inference.ai.azure.com"
    
    return OpenAI(api_key=token, base_url=endpoint)

# History file path
HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history", "team_allocation.json")

# Ensure history directory exists
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

def load_allocation_history() -> List[Dict]:
    """Load team allocation history from file"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_allocation_history(history: List[Dict]):
    """Save team allocation history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_department_teams(db: Session) -> Dict[int, List[int]]:
    """Get mapping of department IDs to team IDs from database"""
    department_teams = {}
    
    # Query all department-team relationships
    results = db.execute(
        select(DepartmentTeam.department_id, DepartmentTeam.team_id)
    ).all()
    
    # Group teams by department
    for dept_id, team_id in results:
        if dept_id not in department_teams:
            department_teams[dept_id] = []
        department_teams[dept_id].append(team_id)
    
    return department_teams

def get_package_departments(db: Session) -> Dict[int, List[int]]:
    """Get mapping of package IDs to department IDs from database"""
    package_departments = {}
    
    # Query tasks associated with packages
    task_packages = db.execute(
        select(TaskPackage.package_id, Task.department_id)
        .join(Task, TaskPackage.task_id == Task.id)
    ).all()
    
    # Group departments by package
    for package_id, dept_id in task_packages:
        if package_id not in package_departments:
            package_departments[package_id] = []
        if dept_id not in package_departments[package_id]:
            package_departments[package_id].append(dept_id)
    
    return package_departments

def get_team_workload(history: List[Dict]) -> Dict[int, int]:
    """Calculate workload for each team based on allocation history"""
    team_workload = {}
    
    for allocation in history:
        if allocation.get("message") == "success":
            for team_id in allocation.get("allocated_teams", []):
                team_workload[team_id] = team_workload.get(team_id, 0) + 1
    
    return team_workload

def is_team_available(team_id: int, project_start: str, project_end: str, history: List[Dict]) -> bool:
    """Check if a team is available for the given project dates"""
    project_start_date = datetime.strptime(project_start, "%Y-%m-%d")
    project_end_date = datetime.strptime(project_end, "%Y-%m-%d")
    
    for allocation in history:
        if allocation.get("message") != "success" or team_id not in allocation.get("allocated_teams", []):
            continue
        
        alloc_start = datetime.strptime(allocation["start"], "%Y-%m-%d")
        alloc_end = datetime.strptime(allocation["end"], "%Y-%m-%d")
        
        # Check if dates overlap and the end dates are the same
        if (project_start_date <= alloc_end and project_end_date >= alloc_start and 
            project_end_date == alloc_end):
            return False
    
    return True

def allocate_teams(project_id: int, package_id: int, start_date: str, end_date: str) -> Dict[str, Any]:
    """Allocate teams to a project based on the package"""
    db = next(get_db())
    history = load_allocation_history()
    
    # Get department-team and package-department mappings
    department_teams = get_department_teams(db)
    package_departments = get_package_departments(db)
    
    # Calculate team workloads for balancing
    team_workload = get_team_workload(history)
    
    # Check if package exists
    if package_id not in package_departments:
        return {
            "project_id": project_id,
            "message": "failed",
            "package_id": package_id,
            "start": start_date,
            "end": end_date,
            "allocated_teams": [],
            "reason": f"Package ID {package_id} not found"
        }
    
    # Get departments for this package
    departments = package_departments[package_id]
    
    allocated_teams = []
    failed_departments = []
    
    # For each department in the package, allocate one team
    for dept_id in departments:
        if dept_id not in department_teams:
            failed_departments.append(dept_id)
            continue
            
        available_teams = department_teams[dept_id]
        
        # Sort teams by workload (least busy first)
        sorted_teams = sorted(available_teams, key=lambda t: team_workload.get(t, 0))
        
        # Try to find an available team
        team_allocated = False
        for team_id in sorted_teams:
            if is_team_available(team_id, start_date, end_date, history):
                allocated_teams.append(team_id)
                team_workload[team_id] = team_workload.get(team_id, 0) + 1
                team_allocated = True
                break
        
        # If no team is available with ideal scheduling, pick the least busy one
        if not team_allocated:
            allocated_teams.append(sorted_teams[0])
            team_workload[sorted_teams[0]] = team_workload.get(sorted_teams[0], 0) + 1
    
    # Create response
    result = {
        "project_id": project_id,
        "message": "success" if len(failed_departments) == 0 else "failed",
        "package_id": package_id,
        "start": start_date,
        "end": end_date,
        "allocated_teams": allocated_teams
    }
    
    # Add failed departments if any
    if failed_departments:
        result["failed_departments"] = failed_departments
    
    # Save allocation to history if successful
    if result["message"] == "success":
        # Save the relationship to database
        for team_id in allocated_teams:
            project_team = ProjectTeam(project_id=project_id, team_id=team_id)
            db.add(project_team)
        
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            result["message"] = "failed"
            result["reason"] = f"Database error: {str(e)}"
    
    # Update history
    history.append(result)
    save_allocation_history(history)
    
    return result

def process_allocation_request(input_data: Dict) -> Dict:
    """Process team allocation request with RAG enhancement"""
    try:
        # Extract input data
        project_id = input_data.get("project_id")
        package_id = input_data.get("package_id")
        start_date = input_data.get("start")
        end_date = input_data.get("end")
        
        # Validate input
        if None in (project_id, package_id, start_date, end_date):
            return {
                "message": "failed",
                "reason": "Missing required fields: project_id, package_id, start, end"
            }
        
        # Perform allocation
        return allocate_teams(project_id, package_id, start_date, end_date)
        
    except Exception as e:
        return {
            "message": "failed",
            "reason": f"Error processing request: {str(e)}"
        }

def main():
    client = get_openai_client()
    
    # System prompt with RAG context
    system_prompt = """You are an AI API that responds in JSON format to allocate teams to projects based on the departments included in a selected project package.

## Team Allocation Rules:
1. Each department has teams associated with it through the DepartmentTeam relationship.
2. Each package includes a set of departments through tasks.
3. When a package is selected for a project, only the departments of the selected package will have teams allocated.
4. Each department in a package should have exactly one team allocated to the project.
5. Team allocation should balance workload across teams.

## Expected Input:
{
    "project_id": int,
    "package_id": int,
    "start": str,  # Start date of project
    "end": str,    # The event day
}

## Expected Output:
{
    "project_id": int,
    "message": "success/failed",
    "package_id": int,
    "start": str,
    "end": str,
    "allocated_teams": list  # List of allocated team IDs
}
"""

    # Sample user input
    user_input = '{"project_id": 1, "package_id": 1, "start": "2025-04-01", "end": "2025-04-05"}'
    
    # Parse user input
    try:
        input_data = json.loads(user_input)
    except json.JSONDecodeError:
        print(json.dumps({"message": "failed", "reason": "Invalid JSON input"}))
        return
    
    # Process allocation request
    result = process_allocation_request(input_data)
    
    # Output result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()