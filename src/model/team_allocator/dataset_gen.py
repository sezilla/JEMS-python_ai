import json
import random
import os
from datetime import datetime, timedelta
import numpy as np

class TeamAllocatorDatasetGenerator:
    def __init__(self, output_path="src/datasets/team_allocator_dataset.json", num_projects=50):
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.output_path = output_path
        self.num_projects = num_projects
        
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
        
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime(2027, 1, 1)  # 2 years span
        
        # Initialize team workload tracking
        self.team_projects = {team_id: [] for dept in self.department_teams.values() for team_id in dept}
        
    def generate_random_date_range(self):
        """Generate a random project date range between 4 months and 2 years"""
        total_days = (self.end_date - self.start_date).days
        start_offset = random.randint(0, total_days - 120)  # Ensure at least 4 months before end date
        
        start_date = self.start_date + timedelta(days=start_offset)
        
        # Duration between 4 months (120 days) and 2 years (730 days)
        max_duration = min(730, (self.end_date - start_date).days)
        min_duration = 120  # 4 months
        
        duration = random.randint(min_duration, max_duration)
        end_date = start_date + timedelta(days=duration)
        
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    def check_schedule_overlap(self, team_id, start_date, end_date):
        """Check if a new project overlaps with existing projects for a team"""
        new_start = datetime.strptime(start_date, "%Y-%m-%d")
        new_end = datetime.strptime(end_date, "%Y-%m-%d")
        
        for proj in self.team_projects[team_id]:
            proj_start = datetime.strptime(proj["start"], "%Y-%m-%d")
            proj_end = datetime.strptime(proj["end"], "%Y-%m-%d")
            
            # Check for exact same end date
            if proj_end == new_end:
                return True
                
            # Check for overlap
            if (new_start <= proj_end and new_end >= proj_start):
                return True
                
        return False
    
    def allocate_team_for_department(self, dept_id, start_date, end_date):
        """Find the best team for a department based on schedule"""
        available_teams = self.department_teams[dept_id]
        best_team = None
        min_overlaps = float('inf')
        
        for team_id in available_teams:
            # Check current workload
            if len(self.team_projects[team_id]) >= 10:
                continue
                
            # Check for same end date conflicts
            if any(datetime.strptime(proj["end"], "%Y-%m-%d") == datetime.strptime(end_date, "%Y-%m-%d") 
                  for proj in self.team_projects[team_id]):
                continue
            
            # Count schedule overlaps
            overlaps = 0
            new_start = datetime.strptime(start_date, "%Y-%m-%d")
            new_end = datetime.strptime(end_date, "%Y-%m-%d")
            
            for proj in self.team_projects[team_id]:
                proj_start = datetime.strptime(proj["start"], "%Y-%m-%d")
                proj_end = datetime.strptime(proj["end"], "%Y-%m-%d")
                
                if (new_start <= proj_end and new_end >= proj_start):
                    overlaps += 1
            
            # Find team with minimum overlaps
            if best_team is None or overlaps < min_overlaps:
                best_team = team_id
                min_overlaps = overlaps
        
        return best_team
    
    def generate_dataset(self):
        """Generate the full dataset of projects with allocated teams"""
        dataset = []
        
        for project_id in range(1, self.num_projects + 1):
            package_id = random.randint(1, 5)
            start_date, end_date = self.generate_random_date_range()
            
            allocated_teams = []
            departments = self.package_departments[package_id]
            
            for dept_id in departments:
                team_id = self.allocate_team_for_department(dept_id, start_date, end_date)
                
                if team_id:
                    allocated_teams.append(team_id)
                    # Update team's project list
                    self.team_projects[team_id].append({
                        "project_id": project_id,
                        "start": start_date,
                        "end": end_date
                    })
            
            # Only add complete projects where all departments have teams allocated
            if len(allocated_teams) == len(departments):
                dataset.append({
                    "project_id": project_id,
                    "package_id": package_id,
                    "start": start_date,
                    "end": end_date,
                    "allocated_teams": sorted(allocated_teams)
                })
        
        # Save to file
        with open(self.output_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        
        print(f"Dataset with {len(dataset)} projects generated and saved to {self.output_path}")
        return dataset

if __name__ == "__main__":
    generator = TeamAllocatorDatasetGenerator(num_projects=150)
    generator.generate_dataset()