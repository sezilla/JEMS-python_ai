import random
import json
import os
from datetime import datetime
from src.models import Task, TaskPackage, DepartmentTeam, ProjectTeam, Project

PROJECT_HISTORY_FILE = "project-history.json"

class EventTeamAllocator:
    def __init__(self):
        self.team_schedules = {}
        self.project_history = self.load_project_history()
        self.allocated_teams = {}  # Store allocated teams by project name

    def load_project_history(self):
        """ Load project history from JSON file at startup. """
        if os.path.exists(PROJECT_HISTORY_FILE):
            try:
                # 🔥 Fix: Check if the file is empty
                if os.path.getsize(PROJECT_HISTORY_FILE) == 0:
                    return []  # If empty, return an empty list
                
                with open(PROJECT_HISTORY_FILE, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print("⚠️ Warning: project-history.json is corrupt. Resetting file.")
                return []  # If JSON is corrupt, return empty history
        return []
    
    def save_project_history(self):
        """ Save the project history to JSON file. """
        try:
            with open(PROJECT_HISTORY_FILE, "w") as file:
                json.dump(self.project_history, file, indent=4)
        except Exception as e:
            print(f"❌ Error saving project history: {e}")

    def get_package_tasks(self, db, package_id):
        tasks = db.query(TaskPackage.task_id).filter(TaskPackage.package_id == package_id).all()
        return [task.task_id for task in tasks]

    def get_department_for_task(self, db, task_id):
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            return task.department_id
        raise ValueError(f"Task with ID {task_id} not found")

    def get_teams_for_department(self, db, department_id):
        teams = db.query(DepartmentTeam.team_id).filter(DepartmentTeam.department_id == department_id).all()
        return [team.team_id for team in teams]

    def is_team_available(self, team_id, start, end):
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        return all(end_dt < s or start_dt > e for s, e in self.team_schedules.get(team_id, []))

    def allocate_teams(self, db, project_name, package_id, start, end):
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')

        if end_dt < start_dt:
            return {"success": False, "error": "End date cannot be before the start date."}

        package_tasks = self.get_package_tasks(db, package_id)
        if not package_tasks:
            return {"success": False, "error": "No tasks found for the given package."}

        departments_needed = {self.get_department_for_task(db, task_id) for task_id in package_tasks}
        if not departments_needed:
            return {"success": False, "error": "No departments found for the given tasks."}

        allocated_teams = {}

        for dept_id in departments_needed:
            department_teams = self.get_teams_for_department(db, dept_id)
            available_teams = [t for t in department_teams if self.is_team_available(t, start, end)]

            if not available_teams:
                print(f"⚠️ Warning: No available teams for department {dept_id}")  # 🔥 Debugging

            if available_teams:
                selected_team = random.choice(available_teams)
                allocated_teams[dept_id] = selected_team
                self.team_schedules.setdefault(selected_team, []).append((start_dt, end_dt))

        if not allocated_teams:
            print("❌ No teams were allocated for this project.")  # 🔥 Debugging
            return {"success": False, "error": "No teams available for allocation."}

        result = {
            "success": True,
            "project_name": project_name,
            "package_id": package_id,
            "start": start,
            "end": end,
            "allocated_teams": list(allocated_teams.values())  # ✅ Convert to list
        }

        self.project_history.append(result)
        self.allocated_teams[project_name] = allocated_teams  # Store allocated teams
        self.save_project_history()

        print(f"✅ Allocated Teams for {project_name}: {result['allocated_teams']}")  # 🔥 Debugging
        return result

    def save_allocated_teams_to_laravel(self, db, project_name, allocated_teams):
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            raise ValueError(f"Project with name '{project_name}' not found")

        for department_id, team_id in allocated_teams.items():
            project_team_entry = ProjectTeam(project_id=project.id, team_id=team_id)
            db.add(project_team_entry)
        db.commit()