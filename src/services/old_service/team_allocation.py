import json
import os
import torch
from datetime import datetime
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.models import Task, TaskPackage, DepartmentTeam, ProjectTeam, Project
from huggingface_hub import login

PROJECT_HISTORY_FILE = "src/history/project-history.json"
# MODEL_PATH = "./distilbert_team_allocator" #local
MODEL_PATH = "sezilla/distilbert_team_allocator"
ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")  # Use environment variable for security
if not ACCESS_TOKEN:
    raise ValueError("❌ Hugging Face token is missing! Set HF_TOKEN in your environment.")

login(token=ACCESS_TOKEN)

class EventTeamAllocator:
    def __init__(self):
        self.team_schedules = {}
        self.project_history = self.load_project_history()
        self.allocated_teams = {}

        # Load the trained model and tokenizer
        print("📥 Loading trained model...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()  # Set model to evaluation mode
        print("✅ Model loaded successfully.")

    def load_project_history(self):
        """ Load project history from JSON file at startup. """
        if os.path.exists(PROJECT_HISTORY_FILE):
            try:
                if os.path.getsize(PROJECT_HISTORY_FILE) == 0:
                    print("⚠️ project-history.json is empty. Initializing with an empty list.")
                    return []
                with open(PROJECT_HISTORY_FILE, "r") as file:
                    data = json.load(file)
                    print(f"📂 Loaded Project History: {data}")  # Debugging
                    return data
            except json.JSONDecodeError:
                print("⚠️ Warning: project-history.json is corrupt. Resetting file.")
                return []
        print("📂 No existing project-history.json found. Creating a new one.")
        return []
    
    def save_project_history(self):
        """ Save the project history to JSON file. """
        try:
            print(f"📂 Saving Project History: {self.project_history}")  # Debugging
            with open(PROJECT_HISTORY_FILE, "w") as file:
                json.dump(self.project_history, file, indent=4)
            print("✅ Project history saved successfully.")  # Debugging
        except Exception as e:
            print(f"❌ Error saving project history: {e}")

    def get_package_tasks(self, db, package_id):
        tasks = db.query(TaskPackage.task_id).filter(TaskPackage.package_id == package_id).all()
        return [task.task_id for task in tasks]

    def get_department_for_task(self, db, task_id):
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            return task.department_id
        raise ValueError(f"❌ Task with ID {task_id} not found")

    def get_teams_for_department(self, db, department_id):
        teams = db.query(DepartmentTeam.team_id).filter(DepartmentTeam.department_id == department_id).all()
        return [team.team_id for team in teams]

    def is_team_available(self, team_id, start, end):
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        return all(end_dt < s or start_dt > e for s, e in self.team_schedules.get(team_id, []))

    def predict_teams(self, project_name, package_id, start, end):
        """ Predict the best teams using the trained DistilBERT model. """
        input_text = f"{project_name} {package_id} {start} {end}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).squeeze().tolist()

        # Sort teams by confidence score (highest first)
        predicted_teams = sorted(
            enumerate(probabilities), key=lambda x: x[1], reverse=True
        )

        print(f"🔮 Predicted Teams with Scores: {predicted_teams}")  # Debugging
        return [team_id for team_id, score in predicted_teams if score > 0.5]

    def allocate_teams(self, db, project_name, package_id, start, end):
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')

        if end_dt < start_dt:
            return {"success": False, "error": "❌ End date cannot be before the start date."}

        package_tasks = self.get_package_tasks(db, package_id)
        if not package_tasks:
            return {"success": False, "error": "❌ No tasks found for the given package."}

        departments_needed = {self.get_department_for_task(db, task_id) for task_id in package_tasks}
        if not departments_needed:
            return {"success": False, "error": "❌ No departments found for the given tasks."}

        allocated_teams = {}

        for dept_id in departments_needed:
            department_teams = self.get_teams_for_department(db, dept_id)
            print(f"🏢 Department {dept_id} Teams: {department_teams}")  # Debugging

            predicted_teams = self.predict_teams(project_name, package_id, start, end)
            available_teams = [t for t in department_teams if t in predicted_teams and self.is_team_available(t, start, end)]

            if not available_teams:
                print(f"⚠️ Warning: No available predicted teams for department {dept_id}")

            if available_teams:
                selected_team = available_teams[0]
                allocated_teams[dept_id] = selected_team
                self.team_schedules.setdefault(selected_team, []).append((start_dt, end_dt))

        if not allocated_teams:
            print("❌ No teams were allocated for this project.")
            return {"success": False, "error": "No teams available for allocation."}

        result = {
            "success": True,
            "project_name": project_name,
            "package_id": package_id,
            "start": start,
            "end": end,
            "allocated_teams": allocated_teams
        }

        print(f"✅ Allocated Teams for {project_name}: {allocated_teams}")  # Debugging

        # 🔥 Ensure project history gets updated before saving
        self.project_history.append(result)
        print(f"📁 Updated Project History: {self.project_history}")  # Debugging

        self.allocated_teams[project_name] = allocated_teams
        self.save_project_history()

        return result

    def save_allocated_teams_to_laravel(self, db, project_name, allocated_teams):
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            raise ValueError(f"❌ Project with name '{project_name}' not found")

        for department_id, team_id in allocated_teams.items():
            project_team_entry = ProjectTeam(project_id=project.id, team_id=team_id)
            db.add(project_team_entry)
        db.commit()
        print(f"✅ Allocated teams saved to Laravel for project: {project_name}")  # Debugging
