import os
import sys
import json
import datetime
from openai import OpenAI
import logging
from typing import List, Dict, Any, Optional, Set
from functools import lru_cache
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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

# Initialize OpenAI client
client = OpenAI(
    api_key=GITHUB_TOKEN,
    base_url="https://models.inference.ai.azure.com"
)

if not client.api_key:
    raise ValueError("GITHUB_TOKEN is not set in environment variables")

# Constants
history_dir = "src/history"
ALLOCATION_HISTORY = os.path.join(history_dir, "team_allocation.json")
DATE_FORMAT = "%Y-%m-%d"

# Ensure history directory exists
os.makedirs(history_dir, exist_ok=True)

# Initialize allocation history file
if not os.path.exists(ALLOCATION_HISTORY):
    with open(ALLOCATION_HISTORY, "w") as f:
        json.dump([], f)


@contextmanager
def get_db_session():
    """Context manager for database sessions with proper cleanup."""
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


@lru_cache(maxsize=128)
def get_package() -> List[Package]:
    """Get all packages with caching."""
    with get_db_session() as session:
        packages = session.query(Package).all()
        # Detach from session to use with cache
        session.expunge_all()
        return packages


@lru_cache(maxsize=128)
def get_department() -> List[Department]:
    """Get all departments with caching."""
    with get_db_session() as session:
        departments = session.query(Department).all()
        session.expunge_all()
        return departments


@lru_cache(maxsize=128)
def get_task() -> List[Task]:
    """Get all tasks with caching."""
    with get_db_session() as session:
        tasks = session.query(Task).all()
        session.expunge_all()
        return tasks


@lru_cache(maxsize=128)
def get_task_package() -> List[TaskPackage]:
    """Get all task packages with caching."""
    with get_db_session() as session:
        task_packages = session.query(TaskPackage).all()
        session.expunge_all()
        return task_packages


@lru_cache(maxsize=128)
def get_department_team() -> List[DepartmentTeam]:
    """Get all department teams with caching."""
    with get_db_session() as session:
        department_teams = session.query(DepartmentTeam).all()
        session.expunge_all()
        return department_teams


def get_db_allocation_history() -> List[TeamAllocation]:
    """Get allocation history from database."""
    with get_db_session() as session:
        allocations = session.query(TeamAllocation).all()
        session.expunge_all()
        return allocations


def convert_to_safe_dict(obj: Any) -> Dict[str, Any]:
    """Convert SQLAlchemy object to safe dictionary with improved error handling."""
    if obj is None:
        return {}
    
    result = {}
    for attr in dir(obj):
        if attr.startswith('_') or callable(getattr(obj, attr, None)):
            continue
            
        try:
            value = getattr(obj, attr)
            if isinstance(value, datetime.date):
                result[attr] = value.strftime(DATE_FORMAT)
            elif isinstance(value, datetime.datetime):
                result[attr] = value.strftime(f"{DATE_FORMAT} %H:%M:%S")
            elif isinstance(value, (int, float, str, bool, type(None))):
                result[attr] = value
            elif isinstance(value, list):
                # Handle list attributes (like allocated_teams)
                result[attr] = [item if isinstance(item, (int, float, str, bool, type(None))) else str(item) for item in value]
        except (AttributeError, TypeError) as e:
            logger.debug(f"Skipping attribute {attr}: {e}")
            continue
    
    return result


def clean_json_response(response_text: str) -> Dict[str, Any]:
    """Clean and parse AI-generated JSON response with enhanced error handling."""
    if not response_text or not response_text.strip():
        return {"success": False, "error": "Empty response from AI"}
    
    try:
        response_text = response_text.strip()
        
        # Handle code blocks
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_content = response_text
        
        # Extract JSON object
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            raise ValueError("No valid JSON object found in response")
        
        json_content = json_content[start_idx:end_idx+1]
        result = json.loads(json_content)
        
        # Validate required fields
        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")
            
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "success": False,
            "error": f"Invalid JSON format: {str(e)}",
            "raw_response": response_text[:500]  # Limit raw response length
        }
    except Exception as e:
        logger.error(f"Response parsing error: {e}")
        return {
            "success": False,
            "error": f"Failed to parse response: {str(e)}",
            "raw_response": response_text[:500]
        }


def get_package_departments(package_id: int) -> List[int]:
    """Get departments associated with a specific package with improved efficiency."""
    logger.info(f"Getting departments for package ID: {package_id}")
    
    try:
        task_packages = get_task_package()
        tasks = get_task()
        
        # Convert to dicts for processing
        task_package_data = [convert_to_safe_dict(tp) for tp in task_packages]
        task_data = [convert_to_safe_dict(t) for t in tasks]
        
        # Create lookup sets for better performance
        package_task_ids = {tp['task_id'] for tp in task_package_data 
                          if tp.get('package_id') == package_id and 'task_id' in tp}
        
        department_ids = {t['department_id'] for t in task_data 
                        if t.get('id') in package_task_ids and 'department_id' in t}
        
        unique_departments = list(department_ids)
        logger.info(f"Package {package_id} requires departments: {unique_departments}")
        return unique_departments
        
    except Exception as e:
        logger.error(f"Error getting package departments: {e}")
        return []


def get_department_teams(department_ids: List[int]) -> List[int]:
    """Get teams associated with given departments with improved efficiency."""
    logger.info(f"Getting teams for departments: {department_ids}")
    
    try:
        department_teams = get_department_team()
        department_team_data = [convert_to_safe_dict(dt) for dt in department_teams]
        
        # Use set for faster lookups
        department_set = set(department_ids)
        team_ids = {dt['team_id'] for dt in department_team_data 
                   if dt.get('department_id') in department_set and 'team_id' in dt}
        
        unique_teams = list(team_ids)
        logger.info(f"Found teams for departments {department_ids}: {unique_teams}")
        return unique_teams
        
    except Exception as e:
        logger.error(f"Error getting department teams: {e}")
        return []


def load_allocation_history() -> List[Dict[str, Any]]:
    """Load allocation history with error handling."""
    try:
        with open(ALLOCATION_HISTORY, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading allocation history: {e}")
        return []


def get_team_availability(team_ids: List[int], start_date: str, end_date: str) -> Dict[int, List[Dict]]:
    """Get availability of teams within specified date range with improved error handling."""
    logger.info(f"Checking availability for teams {team_ids} from {start_date} to {end_date}")
    
    try:
        allocation_data = load_allocation_history()
        team_schedules = {team_id: [] for team_id in team_ids}
        team_set = set(team_ids)
        
        for allocation in allocation_data:
            if not isinstance(allocation, dict) or 'allocated_teams' not in allocation:
                continue
                
            allocated_teams = allocation['allocated_teams']
            if not isinstance(allocated_teams, list):
                continue
            
            # Process each allocated team
            for team_id in allocated_teams:
                if team_id in team_set:
                    schedule_entry = {}
                    for key in ['project_id', 'start', 'end']:
                        if key in allocation:
                            schedule_entry[key] = allocation[key]
                    
                    if schedule_entry:  # Only add if we have some data
                        team_schedules[team_id].append(schedule_entry)
        
        return team_schedules
        
    except Exception as e:
        logger.error(f"Error getting team availability: {e}")
        return {team_id: [] for team_id in team_ids}


def parse_date(date_str: str) -> Optional[datetime.date]:
    """Parse date string with error handling."""
    try:
        return datetime.datetime.strptime(date_str, DATE_FORMAT).date()
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid date format '{date_str}': {e}")
        return None


def check_date_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Check if two date ranges overlap with improved error handling."""
    try:
        start1_date = parse_date(start1)
        end1_date = parse_date(end1)
        start2_date = parse_date(start2)
        end2_date = parse_date(end2)
        
        # If any date parsing failed, assume no overlap for safety
        if None in (start1_date, end1_date, start2_date, end2_date):
            logger.warning(f"Date parsing failed for overlap check: {start1}, {end1}, {start2}, {end2}")
            return False
        
        return max(start1_date, start2_date) <= min(end1_date, end2_date)
        
    except Exception as e:
        logger.error(f"Error checking date overlap: {e}")
        return False


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
    
    logger.info(f"Available teams: {available_team_ids}")
    logger.info(f"Unavailable teams (already allocated): {unavailable_team_ids}")
    
    return available_teams


def count_team_allocations(team_ids: List[int]) -> Dict[int, int]:
    """Count the number of existing allocations for each team."""
    try:
        allocation_data = load_allocation_history()
        allocation_counts = {team_id: 0 for team_id in team_ids}
        team_set = set(team_ids)
        
        for allocation in allocation_data:
            if not isinstance(allocation, dict) or 'allocated_teams' not in allocation:
                continue
                
            allocated_teams = allocation['allocated_teams']
            if isinstance(allocated_teams, list):
                for team_id in allocated_teams:
                    if team_id in team_set:
                        allocation_counts[team_id] += 1
        
        logger.info(f"Team allocation counts: {allocation_counts}")
        return allocation_counts
        
    except Exception as e:
        logger.error(f"Error counting team allocations: {e}")
        return {team_id: 0 for team_id in team_ids}


def build_team_priority_mapping(package_dept_ids: List[int], eligible_team_ids: List[int], 
                              available_teams: Dict[int, bool], allocation_counts: Dict[int, int]) -> Dict[int, List[int]]:
    """Build team priority mapping for each department."""
    try:
        dept_teams = get_department_team()
        dept_team_data = [convert_to_safe_dict(dt) for dt in dept_teams]
        
        # Create team to department mapping
        team_to_dept = {}
        for dt in dept_team_data:
            if 'team_id' in dt and 'department_id' in dt:
                team_to_dept[dt['team_id']] = dt['department_id']
        
        # Group teams by department
        dept_to_teams = {}
        for team_id, dept_id in team_to_dept.items():
            if dept_id in package_dept_ids and team_id in eligible_team_ids:
                if dept_id not in dept_to_teams:
                    dept_to_teams[dept_id] = []
                dept_to_teams[dept_id].append(team_id)
        
        # Sort teams by priority (available first, then by allocation count)
        team_priority = {}
        for dept_id, teams in dept_to_teams.items():
            sorted_teams = sorted(teams, key=lambda team_id: (
                not available_teams.get(team_id, True),  # Available teams first
                allocation_counts.get(team_id, 0)        # Then by allocation count
            ))
            team_priority[dept_id] = sorted_teams
        
        return team_priority, team_to_dept, dept_to_teams
        
    except Exception as e:
        logger.error(f"Error building team priority mapping: {e}")
        return {}, {}, {}


def create_allocation_prompt(team_allocation: TeamAllocationRequest, eligible_team_ids: List[int],
                           available_teams: Dict[int, bool], team_priority: Dict[int, List[int]],
                           allocation_counts: Dict[int, int], package_dept_ids: List[int],
                           dept_names: Dict[int, str], team_to_dept: Dict[int, int],
                           dept_to_teams: Dict[int, List[int]], package_data: List[Dict],
                           team_schedules: Dict[int, List[Dict]]) -> str:
    """Create the allocation prompt for the AI."""
    available_team_ids = [team_id for team_id, available in available_teams.items() if available]
    busy_team_ids = [team_id for team_id, available in available_teams.items() if not available]
    
    return f"""
You are a team allocation assistant. Based on the following data:

PROJECT INFO:
- Project ID: {team_allocation.project_id}
- Package ID: {team_allocation.package_id}
- Time span: {team_allocation.start} to {team_allocation.end}

ALLOCATION RULES:
- CRITICAL: You must ONLY allocate teams from this pre-validated list: {eligible_team_ids}
- ENSURE ALL TEAMS GET ALLOCATED OVER TIME - balance workload across all teams
- Available teams (no timeline conflicts): {available_team_ids}
- Teams with existing workload during timeline: {busy_team_ids}
- Teams ordered by priority (least allocated first): {json.dumps(team_priority)}
- Team allocation counts: {json.dumps(allocation_counts)}
- Always prioritize teams with fewer existing allocations
- Teams with zero allocations should get highest priority
- Each required department must have at least one team allocated
- MOST IMPORTANT: Only allocate 1 team per department
- When all teams are busy, stack projects while maintaining balanced workload

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
        {{"department_name": "department_name2", "team_id": team_id2}}
    ]
}}

REQUIREMENTS:
1. Only include departments that are required for this package
2. Each team_id MUST be from the eligible_team_ids list
3. Prioritize teams with fewer existing allocations
4. Ensure every department has exactly one team allocated
5. Spread workload across different teams when possible
"""


def validate_allocation_result(result: Dict[str, Any], eligible_team_ids: List[int], 
                             package_dept_ids: List[int]) -> List[int]:
    """Validate and extract allocated teams from AI result."""
    allocated_teams = []
    
    if not result.get("success", False):
        logger.warning(f"AI returned unsuccessful result: {result.get('error', 'Unknown error')}")
        return allocated_teams
    
    if "allocated_teams" not in result or not isinstance(result["allocated_teams"], list):
        logger.warning("AI result missing or invalid allocated_teams field")
        return allocated_teams
    
    for team_alloc in result["allocated_teams"]:
        if not isinstance(team_alloc, dict) or "team_id" not in team_alloc:
            logger.warning(f"Invalid team allocation format: {team_alloc}")
            continue
            
        team_id = team_alloc["team_id"]
        if team_id not in eligible_team_ids:
            logger.warning(f"AI attempted to allocate invalid team: {team_id}")
            continue
            
        allocated_teams.append(team_id)
    
    return allocated_teams


def ensure_department_coverage(allocated_teams: List[int], package_dept_ids: List[int],
                             team_priority: Dict[int, List[int]], team_to_dept: Dict[int, int]) -> List[int]:
    """Ensure all required departments have at least one allocated team."""
    # Check which departments are already covered
    allocated_dept_ids = set()
    for team_id in allocated_teams:
        if team_id in team_to_dept:
            allocated_dept_ids.add(team_to_dept[team_id])
    
    # Add teams for uncovered departments
    final_teams = allocated_teams.copy()
    for dept_id in package_dept_ids:
        if dept_id not in allocated_dept_ids and dept_id in team_priority and team_priority[dept_id]:
            least_allocated_team = team_priority[dept_id][0]
            final_teams.append(least_allocated_team)
            logger.info(f"Added team {least_allocated_team} from department {dept_id} to ensure coverage")
    
    return final_teams


def allocate_team(team_allocation: TeamAllocationRequest) -> TeamAllocationResponse:
    """Allocate teams to a project based on package requirements and team availability."""
    logger.info(f"\nAllocating teams for Project ID: {team_allocation.project_id}, Package ID: {team_allocation.package_id}")
    logger.info(f"Project timespan: {team_allocation.start} to {team_allocation.end}")
    
    try:
        # Validate input dates
        if not parse_date(team_allocation.start) or not parse_date(team_allocation.end):
            raise ValueError("Invalid date format in request")
        
        # Get basic data
        packages = get_package()
        departments = get_department()
        package_data = [convert_to_safe_dict(p) for p in packages]
        department_data = [convert_to_safe_dict(d) for d in departments]
        
        # Get package requirements
        package_dept_ids = get_package_departments(team_allocation.package_id)
        if not package_dept_ids:
            raise ValueError(f"No departments found for package {team_allocation.package_id}")
        
        # Get eligible teams
        eligible_team_ids = get_department_teams(package_dept_ids)
        if not eligible_team_ids:
            raise ValueError(f"No teams found for departments {package_dept_ids}")
        
        # Check team availability
        team_schedules = get_team_availability(eligible_team_ids, team_allocation.start, team_allocation.end)
        available_teams = identify_available_teams(team_schedules, team_allocation.start, team_allocation.end)
        allocation_counts = count_team_allocations(eligible_team_ids)
        
        # Build team priority mapping
        team_priority, team_to_dept, dept_to_teams = build_team_priority_mapping(
            package_dept_ids, eligible_team_ids, available_teams, allocation_counts
        )
        
        # Create department names mapping
        dept_names = {d.get('id'): d.get('name', f"Department {d.get('id')}") 
                     for d in department_data if 'id' in d}
        
        # Create prompt and call AI
        prompt = create_allocation_prompt(
            team_allocation, eligible_team_ids, available_teams, team_priority,
            allocation_counts, package_dept_ids, dept_names, team_to_dept,
            dept_to_teams, package_data, team_schedules
        )
        
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
        
        # Process AI response
        output_text = response.choices[0].message.content
        result = clean_json_response(output_text)
        
        if "error" in result:
            raise ValueError(f"AI returned error: {result.get('error')}")
        
        # Validate and extract allocated teams
        allocated_teams = validate_allocation_result(result, eligible_team_ids, package_dept_ids)
        
        # Use fallback allocation if AI failed
        if not allocated_teams:
            logger.warning("AI did not allocate any valid teams. Using fallback allocation.")
            for dept_id in package_dept_ids:
                if dept_id in team_priority and team_priority[dept_id]:
                    allocated_teams.append(team_priority[dept_id][0])
        
        # Ensure all departments are covered
        final_allocated_teams = ensure_department_coverage(
            allocated_teams, package_dept_ids, team_priority, team_to_dept
        )
        
        logger.info(f"Final allocated teams: {final_allocated_teams}")
        
        return TeamAllocationResponse(
            success=True,
            project_id=team_allocation.project_id,
            package_id=team_allocation.package_id,
            start=team_allocation.start,
            end=team_allocation.end,
            allocated_teams=final_allocated_teams
        )
        
    except Exception as e:
        logger.error(f"Allocation failed: {str(e)}")
        raise ValueError(f"Failed to process allocation: {str(e)}")


def sync_db_to_json():
    """Sync database allocation history to the ALLOCATION_HISTORY JSON file."""
    try:
        allocations = get_db_allocation_history()
        
        # Convert to JSON serializable format
        allocation_data = []
        for alloc in allocations:
            allocation_entry = {
                "project_id": alloc.project_id,
                "package_id": alloc.package_id,
                "start": str(alloc.start_date) if alloc.start_date else None,
                "end": str(alloc.end_date) if alloc.end_date else None,
                "allocated_teams": alloc.allocated_teams or []
            }
            allocation_data.append(allocation_entry)
        
        # Write to JSON file
        with open(ALLOCATION_HISTORY, "w") as f:
            json.dump(allocation_data, f, indent=4)
        
        logger.info(f"Synced {len(allocations)} allocations to {ALLOCATION_HISTORY}")
        
    except Exception as e:
        logger.error(f"Error syncing database to JSON: {e}")
        raise


def clear_cache():
    """Clear all cached data."""
    get_package.cache_clear()
    get_department.cache_clear()
    get_task.cache_clear()
    get_task_package.cache_clear()
    get_department_team.cache_clear()
    logger.info("Cache cleared")


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
        print("Generated Team Allocation:")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(json.dumps({
            "success": False, 
            "message": "failed", 
            "reason": str(e)
        }, indent=2))