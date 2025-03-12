import json
import random
import datetime
from typing import List, Dict, Any
import os
from collections import defaultdict

class TeamAllocationDatasetGenerator:
    def __init__(
        self,
        num_projects: int = 500,
        min_project_span_months: int = 6,
        max_project_span_months: int = 18,
        timeline_years: int = 4,
        max_overlapping_projects: int = 5,
        output_path: str = "src/datasets/team_allocation_dataset.json"
    ):
        """
        Initialize the dataset generator for team allocation simulation.
        
        Parameters:
        -----------
        num_projects : int
            Number of projects to generate
        min_project_span_months : int
            Minimum project duration in months
        max_project_span_months : int
            Maximum project duration in months
        timeline_years : int
            Number of years the dataset should span
        max_overlapping_projects : int
            Maximum number of projects that can overlap at any given time
        output_path : str
            Path to save the generated dataset
        """
        self.num_projects = num_projects
        self.min_project_span_months = min_project_span_months
        self.max_project_span_months = max_project_span_months
        self.timeline_years = timeline_years
        self.max_overlapping_projects = max_overlapping_projects
        self.output_path = output_path
        
        # Department teams configuration
        self.department_teams = {
            1: [1, 2, 3, 4, 5, 6],          # Catering
            2: [7, 8, 9, 10, 11, 12],       # Hair and Makeup
            3: [13, 14, 15, 16, 17, 18],    # Photo and Video
            4: [19, 20, 21, 22, 23, 24],    # Designing
            5: [25, 26, 27, 28, 29, 30],    # Entertainment
            6: [31, 32, 33, 34, 35, 36],    # Coordination
        }
        
        # Package departments configuration
        self.package_departments = {
            1: [1, 2, 4, 5, 6],             # Ruby Package
            2: [1, 2, 3, 4, 5, 6],          # Garnet Package
            3: [1, 2, 3, 4, 5, 6],          # Emerald Package
            4: [1, 2, 3, 4, 5, 6],          # Infinity Package
            5: [1, 2, 3, 4, 5, 6],          # Sapphire Package
        }
        
        # Timeline boundaries
        self.start_date = datetime.date(2020, 1, 1)
        self.end_date = self.start_date + datetime.timedelta(days=365 * self.timeline_years)
        
        # Track team allocations for load balancing
        self.team_usage_counter = {team_id: 0 for dept in self.department_teams.values() for team_id in dept}
        
        # Track active projects by date for overlapping constraint
        self.active_projects_by_date = defaultdict(int)

    def random_date_in_range(self) -> datetime.date:
        """Generate a random date within the timeline range."""
        time_delta = self.end_date - self.start_date
        random_days = random.randint(0, time_delta.days)
        return self.start_date + datetime.timedelta(days=random_days)
    
    def generate_project_timespan(self, start_date: datetime.date) -> tuple:
        """Generate project end date based on the start date and duration constraints."""
        # Convert months to days (approximate)
        min_days = int(self.min_project_span_months * 30.44)  # Average days per month
        max_days = int(self.max_project_span_months * 30.44)
        
        duration_days = random.randint(min_days, max_days)
        end_date = start_date + datetime.timedelta(days=duration_days)
        
        # Ensure end_date doesn't exceed the overall timeline
        if end_date > self.end_date:
            end_date = self.end_date
            
        return start_date, end_date
    
    def check_overlapping_constraint(self, start_date: datetime.date, end_date: datetime.date) -> bool:
        """Check if adding a project in the given date range would exceed the max overlapping projects constraint."""
        # Sample several dates in the range to check constraints
        # (For simplicity, we'll check weekly intervals)
        current_date = start_date
        while current_date <= end_date:
            if self.active_projects_by_date[current_date] >= self.max_overlapping_projects:
                return False
            current_date += datetime.timedelta(days=7)
        return True
    
    def update_active_projects(self, start_date: datetime.date, end_date: datetime.date):
        """Update the active projects counter for a new project."""
        current_date = start_date
        while current_date <= end_date:
            self.active_projects_by_date[current_date] += 1
            current_date += datetime.timedelta(days=7)
    
    def select_teams_for_package(self, package_id: int) -> List[int]:
        """Select teams based on the package requirements and load balance."""
        departments = self.package_departments[package_id]
        selected_teams = []
        
        for dept_id in departments:
            available_teams = self.department_teams[dept_id]
            # Choose the team with the least allocations for load balancing
            team_candidates = [(team_id, self.team_usage_counter[team_id]) for team_id in available_teams]
            team_candidates.sort(key=lambda x: x[1])  # Sort by usage count
            
            selected_team = team_candidates[0][0]  # Select team with lowest usage
            selected_teams.append(selected_team)
            self.team_usage_counter[selected_team] += 1
            
        return selected_teams
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the complete dataset of projects."""
        dataset = []
        
        for project_id in range(1, self.num_projects + 1):
            # Try to find a valid date range within constraints
            attempts = 0
            valid_dates_found = False
            
            while attempts < 50 and not valid_dates_found:
                potential_start_date = self.random_date_in_range()
                potential_start, potential_end = self.generate_project_timespan(potential_start_date)
                
                if self.check_overlapping_constraint(potential_start, potential_end):
                    valid_dates_found = True
                    start_date, end_date = potential_start, potential_end
                    self.update_active_projects(start_date, end_date)
                else:
                    attempts += 1
            
            if not valid_dates_found:
                print(f"Warning: Could not find valid dates for project {project_id} after {attempts} attempts")
                continue
            
            # Select package and teams
            package_id = random.randint(1, len(self.package_departments))
            allocated_teams = self.select_teams_for_package(package_id)
            
            # Create project entry
            project = {
                "project_id": project_id,
                # "status": "success",  # All projects are marked as successful in this dataset
                "package_id": package_id,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "allocated_teams": allocated_teams
            }
            
            dataset.append(project)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]]):
        """Save the generated dataset to the specified output path."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        
        print(f"Dataset with {len(dataset)} projects saved to {self.output_path}")
    
    def generate_and_save(self):
        """Generate and save the dataset in one operation."""
        dataset = self.generate_dataset()
        self.save_dataset(dataset)
        return dataset

# Example usage with default parameters
if __name__ == "__main__":
    generator = TeamAllocationDatasetGenerator(
        num_projects=200,
        min_project_span_months=6,
        max_project_span_months=18,
        timeline_years=4,
        max_overlapping_projects=5,
        output_path="src/datasets/team_allocation_dataset.json"
    )
    
    dataset = generator.generate_and_save()
    
    # Print summary statistics
    teams_allocation_counts = generator.team_usage_counter
    print("\nTeam allocation summary:")
    for dept_id, teams in generator.department_teams.items():
        dept_name = ["Catering", "Hair and Makeup", "Photo and Video", 
                    "Designing", "Entertainment", "Coordination"][dept_id-1]
        print(f"\nDepartment {dept_id} ({dept_name}):")
        for team_id in teams:
            print(f"  Team {team_id}: {teams_allocation_counts[team_id]} projects")
    
    # Calculate average project duration
    total_days = 0
    for project in dataset:
        start = datetime.datetime.strptime(project["start"], "%Y-%m-%d").date()
        end = datetime.datetime.strptime(project["end"], "%Y-%m-%d").date()
        days = (end - start).days
        total_days += days
    
    avg_duration = total_days / len(dataset)
    print(f"\nAverage project duration: {avg_duration:.2f} days ({avg_duration/30.44:.2f} months)")
    
    # Package distribution
    package_counts = {}
    for project in dataset:
        package_id = project["package_id"]
        package_counts[package_id] = package_counts.get(package_id, 0) + 1
    
    print("\nPackage distribution:")
    package_names = {1: "Ruby", 2: "Garnet", 3: "Emerald", 4: "Infinity", 5: "Sapphire"}
    for package_id, count in package_counts.items():
        print(f"  {package_names[package_id]} Package: {count} projects ({count/len(dataset)*100:.2f}%)")