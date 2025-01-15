import random
from datetime import datetime
from src.models import Task, TaskPackage, DepartmentTeam, ProjectTeam, Project

class EventTeamAllocator:
    def __init__(self):
        self.team_schedules = {}
        self.project_history = []
        self.allocated_teams = {}  # Store allocated teams by project name

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

        package_tasks = self.get_package_tasks(db, package_id)
        departments_needed = {self.get_department_for_task(db, task_id) for task_id in package_tasks}

        allocated_teams = {}

        for dept_id in departments_needed:
            department_teams = self.get_teams_for_department(db, dept_id)
            available_teams = [t for t in department_teams if self.is_team_available(t, start, end)]

            if not available_teams:
                continue

            selected_team = random.choice(available_teams)
            allocated_teams[dept_id] = selected_team
            self.team_schedules.setdefault(selected_team, []).append((start_dt, end_dt))

        self.save_allocated_teams_to_laravel(db, project_name, allocated_teams)

        result = {
            'project_name': project_name,
            'package_id': package_id,
            'start': start,
            'end': end,
            'allocated_teams': allocated_teams
        }
        self.project_history.append(result)
        self.allocated_teams[project_name] = allocated_teams  # Store allocated teams
        return result

    def save_allocated_teams_to_laravel(self, db, project_name, allocated_teams):
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            raise ValueError(f"Project with name '{project_name}' not found")

        for department_id, team_id in allocated_teams.items():
            project_team_entry = ProjectTeam(project_id=project.id, team_id=team_id)
            db.add(project_team_entry)
        db.commit()