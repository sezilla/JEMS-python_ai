import json
import random
from datetime import datetime, timedelta
import os
from collections import defaultdict

# Define necessary dictionaries for team structure
department_teams = {
    1: [1, 2, 3, 4, 5, 6],          # Catering
    2: [7, 8, 9, 10, 11, 12],       # Hair and Makeup
    3: [13, 14, 15, 16, 17, 18],    # Photo and Video
    4: [19, 20, 21, 22, 23, 24],    # Designing
    5: [25, 26, 27, 28, 29, 30],    # Entertainment
    6: [31, 32, 33, 34, 35, 36],    # Coordination
}

package_departments = {
    1: [1, 2, 3, 4, 5],                 # Basic Package
    2: [1, 2, 3, 4, 5, 6],              # Standard Package
    3: [1, 2, 3, 4, 5, 6],              # Premium Package
    4: [1, 2, 3, 4, 5, 6],              # Luxury Package
    5: [1, 2, 3, 4, 5, 6],              # Custom Package
}

# Function to generate random project names
def random_project_name():
    first_names = ["Rigel", "Vega", "Sirius", "Altair", "Polaris", "Luna", "Nova", "Orion", "Aurora", 
                  "Lyra", "Capella", "Bellatrix", "Aldebaran", "Betelgeuse", "Deneb", "Antares", 
                  "Arcturus", "Regulus", "Spica", "Procyon", "Castor", "Pollux", "Albireo", "Mira"]
    last_names = ["Event", "Wedding", "Campaign", "Gala", "Summit", "Showcase", "Expo", "Festival", 
                 "Conference", "Seminar", "Workshop", "Retreat", "Meetup", "Symposium", "Concert", 
                 "Performance", "Recital", "Gathering", "Rally", "Celebration", "Party", "Reception"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Function to check team availability
def is_team_available(team_id, start_date, end_date, team_schedules):
    for scheduled_start, scheduled_end in team_schedules.get(team_id, []):
        # Check if there's an overlap in schedules
        if not (end_date < scheduled_start or start_date > scheduled_end):
            return False
    return True

# Function to get best available team from department
def get_best_available_team(department_id, start_date, end_date, team_schedules):
    available_teams = []
    
    for team_id in department_teams[department_id]:
        if is_team_available(team_id, start_date, end_date, team_schedules):
            # Calculate team load (number of projects assigned)
            team_load = len(team_schedules.get(team_id, []))
            available_teams.append((team_id, team_load))
    
    if available_teams:
        # Sort by team load (prefer teams with fewer assignments)
        available_teams.sort(key=lambda x: x[1])
        return available_teams[0][0]
    
    # If no team is fully available, find team with least overlap
    department_team_ids = department_teams[department_id]
    if not department_team_ids:
        return None
        
    # Fall back to team with fewest assignments
    team_loads = [(team_id, len(team_schedules.get(team_id, []))) for team_id in department_team_ids]
    team_loads.sort(key=lambda x: x[1])
    return team_loads[0][0]

# Ensure directory exists
os.makedirs("src/datasets", exist_ok=True)

# Generate dataset
num_samples = 600  # Increased sample size for better training
team_schedules = defaultdict(list)
dataset = []

# Generate projects spanning 2 years
start_range = datetime(2025, 1, 1)
end_range = datetime(2029, 1, 1)

for _ in range(num_samples):
    package_id = random.randint(1, 5)
    
    # Random project duration between 4 months and 2 years
    min_duration = 120  # 4 months in days
    max_duration = 730  # 2 years in days
    
    # Generate random start date
    days_range = (end_range - start_range).days - max_duration
    if days_range <= 0:
        days_range = 1
    random_days = random.randint(0, days_range)
    start_date = start_range + timedelta(days=random_days)
    
    # Generate random duration and end date
    duration = random.randint(min_duration, max_duration)
    end_date = start_date + timedelta(days=duration)
    
    # Get the departments for the selected package
    departments = package_departments[package_id]
    
    # Allocate teams from each department
    allocated_teams = []
    for dept_id in departments:
        team_id = get_best_available_team(dept_id, start_date, end_date, team_schedules)
        if team_id:
            allocated_teams.append(team_id)
            team_schedules[team_id].append((start_date, end_date))
    
    # Create project entry
    project = {
        "project_name": random_project_name(),
        "package_id": package_id,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "allocated_teams": allocated_teams
    }
    dataset.append(project)
    
    # Print progress
    if (_ + 1) % 50 == 0:
        print(f"Generated {_ + 1}/{num_samples} projects")

# Save dataset to JSON
dataset_path = "src/datasets/team_allocator_dataset.json"
with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset generation complete. Saved to {dataset_path}")
print(f"Generated {len(dataset)} projects with realistic team allocations")

# Print dataset statistics
team_usage = {team_id: len(schedules) for team_id, schedules in team_schedules.items()}
most_used_team = max(team_usage.items(), key=lambda x: x[1])
least_used_team = min(team_usage.items(), key=lambda x: x[1])

print(f"Most utilized team: Team {most_used_team[0]} with {most_used_team[1]} projects")
print(f"Least utilized team: Team {least_used_team[0]} with {least_used_team[1]} projects")
print(f"Average projects per team: {sum(team_usage.values()) / len(team_usage):.2f}")