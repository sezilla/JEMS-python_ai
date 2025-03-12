import sys
import os
from datetime import datetime

# Ensure the correct module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.services.team_allocation import TeamAllocatorService

def main():
    """Test Team Allocation with minimal input."""
    # Ensure we use local models if available
    use_local = os.path.exists("src/classifier/baseModel/JEMS_team_allocator/")
    service = TeamAllocatorService(use_local_model=use_local)
    
    # Get user inputs
    try:
        project_id = int(input("Enter project ID: "))
        package_id = int(input("Enter package ID (1-5): "))
        start_date = input("Enter project start date (YYYY-MM-DD): ")
        end_date = input("Enter project end date (YYYY-MM-DD): ")
        
        # Validate dates
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid input. Ensure correct numeric IDs and date format YYYY-MM-DD.")
        return
    
    # Run team allocation
    print("\nAllocating teams...")
    result = service.allocate_teams(project_id, package_id, start_date, end_date)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
        if "missing_departments" in result:
            print(f"Missing departments: {', '.join(result['missing_departments'])}")
    else:
        print("\nAllocation successful!")
        for dept_id, allocation in result['department_allocations'].items():
            print(f"- {allocation['department_name']}: Team {allocation['team_id']} (Confidence: {allocation['confidence']:.2f})")

if __name__ == "__main__":
    main()
