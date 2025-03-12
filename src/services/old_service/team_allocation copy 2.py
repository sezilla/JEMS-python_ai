import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

class TeamAllocatorService:
    def __init__(self, use_local_model=True, model_path="sezilla/team_allocator"):
        """
        Initialize the Team Allocator Service
        
        Args:
            use_local_model (bool): Whether to use locally stored models or fetch from HuggingFace
            model_path (str): Path to remote model repository if not using local models
        """
        self.use_local_model = use_local_model
        self.remote_model_path = model_path
        
        self.base_model_path = "src/classifier/baseModel/JEMS_team_allocator/"
        self.history_path = "src/classifier/history/team_allocation.json"
        
        # Ensure directories exist
        os.makedirs(self.base_model_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        
        # Department teams configuration
        self.department_teams = {
            1: [1, 2, 3, 4, 5, 6],          # Catering
            2: [7, 8, 9, 10, 11, 12],       # Hair and Makeup
            3: [13, 14, 15, 16, 17, 18],    # Photo and Video
            4: [19, 20, 21, 22, 23, 24],    # Designing
            5: [25, 26, 27, 28, 29, 30],    # Entertainment
            6: [31, 32, 33, 34, 35, 36],    # Coordination
        }
        
        # Department names for better readability
        self.department_names = {
            1: "Catering",
            2: "Hair and Makeup",
            3: "Photo and Video",
            4: "Designing",
            5: "Entertainment",
            6: "Coordination"
        }
        
        # Package departments configuration
        self.package_departments = {
            1: [1, 2, 4, 5, 6],             # Ruby Package
            2: [1, 2, 3, 4, 5, 6],          # Garnet Package
            3: [1, 2, 3, 4, 5, 6],          # Emerald Package
            4: [1, 2, 3, 4, 5, 6],          # Infinity Package
            5: [1, 2, 3, 4, 5, 6],          # Sapphire Package
        }
        
        # Package names for better readability
        self.package_names = {
            1: "Ruby Package",
            2: "Garnet Package",
            3: "Emerald Package",
            4: "Infinity Package",
            5: "Sapphire Package"
        }
        
        # Initialize RAG components
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.rag_data = []
        
        # Load department models
        self.department_models = {}
        self.load_models()
        
    def load_models(self):
        """Load all department models and RAG components"""
        print("Loading models and RAG components...")
        
        # Load models for each department
        for dept_id in self.department_teams:
            if self.use_local_model:
                model_path = os.path.join(self.base_model_path, f"department_{dept_id}_model.pt")
                metadata_path = os.path.join(self.base_model_path, f"department_{dept_id}_metadata.json")
                
                # Check if model exists
                if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                    print(f"Local model for Department {dept_id} not found")
                    continue
                    
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Initialize model architecture
                model = nn.Sequential(
                    nn.Linear(11, 256),  # 11 features (including RAG enhancement)
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, len(self.department_teams[dept_id]))
                )
                
                # Load model weights
                try:
                    model.load_state_dict(torch.load(model_path))
                    model.eval()  # Set to evaluation mode
                    self.department_models[dept_id] = {
                        "model": model,
                        "teams": metadata["teams"]
                    }
                    print(f"Successfully loaded model for Department {dept_id}")
                except Exception as e:
                    print(f"Error loading model for Department {dept_id}: {e}")
            else:
                # Load model from HuggingFace
                try:
                    model_file = hf_hub_download(repo_id=self.remote_model_path, 
                                                filename=f"department_{dept_id}_model.pt")
                    metadata_file = hf_hub_download(repo_id=self.remote_model_path, 
                                                  filename=f"department_{dept_id}_metadata.json")
                    
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Initialize model architecture (same as local)
                    model = nn.Sequential(
                        nn.Linear(11, 256),  # 11 features
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, len(self.department_teams[dept_id]))
                    )
                    
                    # Load model weights
                    model.load_state_dict(torch.load(model_file))
                    model.eval()  # Set to evaluation mode
                    self.department_models[dept_id] = {
                        "model": model,
                        "teams": metadata["teams"]
                    }
                    print(f"Successfully loaded model for Department {dept_id} from HuggingFace")
                except Exception as e:
                    print(f"Error loading remote model for Department {dept_id}: {e}")
        
        # Load RAG components
        if self.use_local_model:
            rag_index_path = os.path.join(self.base_model_path, "rag_index.faiss")
            rag_data_path = os.path.join(self.base_model_path, "rag_data.json")
            
            if os.path.exists(rag_index_path) and os.path.exists(rag_data_path):
                try:
                    self.index = faiss.read_index(rag_index_path)
                    with open(rag_data_path, 'r') as f:
                        self.rag_data = json.load(f)
                    print(f"RAG components loaded with {len(self.rag_data)} documents")
                except Exception as e:
                    print(f"Error loading RAG components: {e}")
            else:
                print("RAG components not found locally")
        else:
            # Load RAG components from HuggingFace
            try:
                index_file = hf_hub_download(repo_id=self.remote_model_path, filename="rag_index.faiss")
                data_file = hf_hub_download(repo_id=self.remote_model_path, filename="rag_data.json")
                
                self.index = faiss.read_index(index_file)
                with open(data_file, 'r') as f:
                    self.rag_data = json.load(f)
                print(f"RAG components loaded from HuggingFace with {len(self.rag_data)} documents")
            except Exception as e:
                print(f"Error loading remote RAG components: {e}")
        
        print("Model loading complete")
    
    def _retrieve_similar_projects(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar projects for RAG enhancement
        
        Args:
            query (str): Query string representing the project
            k (int): Number of similar projects to return
            
        Returns:
            List[Dict[str, Any]]: List of similar project records
        """
        if self.index is None or not self.rag_data:
            print("RAG index not available")
            return []
            
        query_embedding = self.sentence_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        retrieved = [self.rag_data[idx] for idx in indices[0]]
        return retrieved
    
    def _get_team_availability(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract team availability from historical data
        
        Returns:
            Dict[int, List[Dict[str, Any]]]: Dictionary of team IDs to their assigned projects
        """
        team_projects = {team_id: [] for dept in self.department_teams.values() for team_id in dept}
        
        # Load historical data
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
                
                # Process history to extract team schedules
                for record in history:
                    if "allocated_teams" in record and "start" in record and "end" in record:
                        for team_id in record["allocated_teams"]:
                            team_projects[team_id].append({
                                "project_id": record.get("project_id", 0),
                                "start": record["start"],
                                "end": record["end"]
                            })
            except Exception as e:
                print(f"Error loading team availability from history: {e}")
        
        return team_projects
    
    def _check_team_availability(self, team_id: int, start_date: str, 
                                end_date: str, team_projects: Dict[int, List[Dict[str, Any]]]) -> Tuple[bool, int]:
        """
        Check if a team is available for a project in the given date range
        
        Args:
            team_id (int): Team ID to check
            start_date (str): Project start date in YYYY-MM-DD format
            end_date (str): Project end date in YYYY-MM-DD format
            team_projects (Dict[int, List[Dict[str, Any]]]): Team assignments dictionary
            
        Returns:
            Tuple[bool, int]: (availability status, number of overlapping projects)
        """
        new_start = datetime.strptime(start_date, "%Y-%m-%d")
        new_end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Count overlaps and check for end date conflicts
        overlaps = 0
        has_end_conflict = False
        
        for proj in team_projects.get(team_id, []):
            proj_start = datetime.strptime(proj["start"], "%Y-%m-%d")
            proj_end = datetime.strptime(proj["end"], "%Y-%m-%d")
            
            # Check for same end date
            if proj_end.date() == new_end.date():
                has_end_conflict = True
            
            # Check for overlap
            if (new_start <= proj_end and new_end >= proj_start):
                overlaps += 1
        
        # Team is not available if it has 10 or more projects or has end date conflict
        if len(team_projects.get(team_id, [])) >= 10 or has_end_conflict:
            return False, overlaps
        
        return True, overlaps
    
    def _get_department_name(self, dept_id: int) -> str:
        """Get department name from ID"""
        return self.department_names.get(dept_id, f"Department {dept_id}")
    
    def _get_package_name(self, package_id: int) -> str:
        """Get package name from ID"""
        return self.package_names.get(package_id, f"Package {package_id}")
    
    def allocate_teams(self, project_id: int, package_id: int, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Allocate teams for a project based on model predictions and availability
        
        Args:
            project_id (int): Project ID
            package_id (int): Package ID
            start_date (str): Project start date in YYYY-MM-DD format
            end_date (str): Project end date in YYYY-MM-DD format
            
        Returns:
            Dict[str, Any]: Allocation results including allocated teams or error message
        """
        print(f"\nProcessing allocation request for Project {project_id}, Package {package_id}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Validate inputs
        if package_id not in self.package_departments:
            return {"error": f"Invalid package_id: {package_id}"}
            
        # Convert dates to datetime objects
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            project_duration = (end_dt - start_dt).days
            
            if project_duration < 120:  # Less than 4 months
                return {"error": "Project duration must be at least 4 months"}
            if project_duration > 730:  # More than 2 years
                return {"error": "Project duration must not exceed 2 years"}
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        
        # RAG enhancement - retrieve similar projects
        query = f"Project with package {package_id} from {start_date} to {end_date}"
        similar_projects = self._retrieve_similar_projects(query)
        
        if similar_projects:
            print(f"Retrieved {len(similar_projects)} similar projects for RAG")
        
        # Get team availability from history
        team_projects = self._get_team_availability()
        
        # Prepare feature vector
        base_features = [
            float(project_id),
            float(package_id),
            float(start_dt.year),
            float(start_dt.month),
            float(start_dt.day),
            float(end_dt.year),
            float(end_dt.month),
            float(end_dt.day),
            float(project_duration)
        ]
        
        # Determine departments for this package
        departments = self.package_departments[package_id]
        
        allocated_teams = []
        confidence_scores = {}
        department_allocations = {}
        
        # Make predictions for each department
        for dept_id in departments:
            if dept_id not in self.department_models:
                print(f"No model available for Department {dept_id}")
                continue
            
            model_info = self.department_models[dept_id]
            model = model_info["model"]
            available_teams = model_info["teams"]
            
            # RAG enhancement from similar projects
            rag_enhancement = 0.0
            if similar_projects:
                for similar_proj in similar_projects:
                    if dept_id in [d for p_id in [similar_proj["package_id"]] for d in self.package_departments.get(p_id, [])]:
                        # Find team from this department in the similar project
                        for team_id in similar_proj.get("allocated_teams", []):
                            if team_id in self.department_teams.get(dept_id, []):
                                rag_enhancement = 1.0  # Simple binary enhancement
                                break
            
            # Prepare feature vector with department ID and RAG enhancement
            features_tensor = torch.tensor([base_features + [float(dept_id), rag_enhancement]], dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                output = model(features_tensor)
                scores = F.softmax(output, dim=1).squeeze().tolist()
            
            # Map scores to team IDs and sort by score (highest first)
            team_scores = [(available_teams[i], score) for i, score in enumerate(scores)]
            team_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Check availability and select the best available team
            team_assigned = False
            for team_id, score in team_scores:
                available, overlaps = self._check_team_availability(team_id, start_date, end_date, team_projects)
                
                if available:
                    allocated_teams.append(team_id)
                    confidence_scores[team_id] = float(score)
                    department_allocations[dept_id] = {
                        "team_id": team_id,
                        "confidence": float(score),
                        "department_name": self._get_department_name(dept_id)
                    }
                    team_assigned = True
                    print(f"Department {dept_id}: Allocated Team {team_id} with confidence {score:.4f}")
                    break
            
            if not team_assigned:
                print(f"Warning: No available team found for Department {dept_id}")
        
        # If not all required departments have a team allocated, return an error
        if len(allocated_teams) != len(departments):
            missing_departments = [self._get_department_name(dept_id) for dept_id in departments 
                                  if dept_id not in department_allocations]
            return {
                "error": "Could not allocate teams for all required departments",
                "missing_departments": missing_departments,
                "allocated_teams": allocated_teams,
                "department_allocations": department_allocations
            }
        
        # Save allocation to history
        allocation_record = {
            "project_id": project_id,
            "package_id": package_id,
            "package_name": self._get_package_name(package_id),
            "start": start_date,
            "end": end_date,
            "allocated_teams": allocated_teams,
            "department_allocations": department_allocations,
            "confidence_scores": confidence_scores,
            "allocation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update history file
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        history.append(allocation_record)
        
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Return successful allocation
        return {
            "allocated_teams": allocated_teams,
            "department_allocations": department_allocations,
            "confidence_scores": confidence_scores,
            "message": "Teams successfully allocated"
        }
    
    def update_models(self) -> Dict[str, Any]:
        """
        Update all models by triggering the trainer
        
        Returns:
            Dict[str, Any]: Status of model update operation
        """
        try:
            from src.model.team_allocator.trainer import TeamAllocatorTrainer
            trainer = TeamAllocatorTrainer()
            success = trainer.train()
            return {"success": success, "message": "Models updated successfully" if success else "Model update failed"}
        except Exception as e:
            return {"success": False, "message": f"Error updating models: {str(e)}"}
    
    def get_allocation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get allocation history records
        
        Args:
            limit (int): Maximum number of history records to return
            
        Returns:
            List[Dict[str, Any]]: List of allocation history records
        """
        if not os.path.exists(self.history_path):
            return []
            
        try:
            with open(self.history_path, 'r') as f:
                history = json.load(f)
            
            # Return most recent records first
            return sorted(history, key=lambda x: x.get("allocation_date", ""), reverse=True)[:limit]
        except Exception as e:
            print(f"Error reading allocation history: {e}")
            return []
    
    def get_team_workload(self) -> Dict[str, Any]:
        """
        Get current workload for all teams
        
        Returns:
            Dict[str, Any]: Workload information organized by department
        """
        team_workload = {team_id: 0 for dept in self.department_teams.values() for team_id in dept}
        
        # Load historical data
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
                
                # Count active projects for each team
                now = datetime.now()
                for record in history:
                    if "allocated_teams" in record and "start" in record and "end" in record:
                        start = datetime.strptime(record["start"], "%Y-%m-%d")
                        end = datetime.strptime(record["end"], "%Y-%m-%d")
                        
                        # Check if project is active
                        if start <= now <= end:
                            for team_id in record["allocated_teams"]:
                                if team_id in team_workload:
                                    team_workload[team_id] += 1
            except Exception as e:
                print(f"Error loading team workload data: {e}")
        
        # Organize by department
        dept_workload = {}
        for dept_id, teams in self.department_teams.items():
            dept_workload[dept_id] = {
                "department_name": self._get_department_name(dept_id),
                "teams": {team_id: {
                    "active_projects": team_workload[team_id],
                    "availability": "High" if team_workload[team_id] < 3 else
                                   "Medium" if team_workload[team_id] < 7 else "Low"
                } for team_id in teams}
            }
        
        return {
            "department_workload": dept_workload,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_package_info(self, package_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about available packages
        
        Args:
            package_id (Optional[int]): Specific package ID to get info for, or None for all packages
            
        Returns:
            Dict[str, Any]: Package information
        """
        if package_id is not None:
            if package_id not in self.package_departments:
                return {"error": f"Invalid package_id: {package_id}"}
                
            departments = [
                {
                    "department_id": dept_id,
                    "name": self._get_department_name(dept_id)
                } for dept_id in self.package_departments[package_id]
            ]
            
            return {
                "package_id": package_id,
                "name": self._get_package_name(package_id),
                "departments": departments
            }
        else:
            # Return info for all packages
            packages = []
            for pkg_id in self.package_departments:
                departments = [
                    {
                        "department_id": dept_id,
                        "name": self._get_department_name(dept_id)
                    } for dept_id in self.package_departments[pkg_id]
                ]
                
                packages.append({
                    "package_id": pkg_id,
                    "name": self._get_package_name(pkg_id),
                    "departments": departments
                })
            
            return {"packages": packages}
    
    def get_department_info(self, dept_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about departments and their teams
        
        Args:
            dept_id (Optional[int]): Specific department ID to get info for, or None for all departments
            
        Returns:
            Dict[str, Any]: Department information
        """
        if dept_id is not None:
            if dept_id not in self.department_teams:
                return {"error": f"Invalid department_id: {dept_id}"}
                
            return {
                "department_id": dept_id,
                "name": self._get_department_name(dept_id),
                "teams": self.department_teams[dept_id]
            }
        else:
            # Return info for all departments
            departments = []
            for d_id in self.department_teams:
                departments.append({
                    "department_id": d_id,
                    "name": self._get_department_name(d_id),
                    "teams": self.department_teams[d_id]
                })
            
            return {"departments": departments}
    
    def get_team_schedule(self, team_id: int) -> Dict[str, Any]:
        """
        Get schedule for a specific team
        
        Args:
            team_id (int): Team ID to get schedule for
            
        Returns:
            Dict[str, Any]: Team schedule information
        """
        # Check if team ID is valid
        team_found = False
        dept_id = None
        
        for d_id, teams in self.department_teams.items():
            if team_id in teams:
                team_found = True
                dept_id = d_id
                break
                
        if not team_found:
            return {"error": f"Invalid team_id: {team_id}"}
        
        # Get team projects
        projects = []
        
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    history = json.load(f)
                
                for record in history:
                    if "allocated_teams" in record and team_id in record["allocated_teams"]:
                        projects.append({
                            "project_id": record["project_id"],
                            "package_id": record["package_id"],
                            "package_name": record.get("package_name", self._get_package_name(record["package_id"])),
                            "start": record["start"],
                            "end": record["end"],
                            "status": self._get_project_status(record["start"], record["end"])
                        })
            except Exception as e:
                print(f"Error loading team schedule data: {e}")
        
        return {
            "team_id": team_id,
            "department_id": dept_id,
            "department_name": self._get_department_name(dept_id),
            "projects": sorted(projects, key=lambda x: x["start"], reverse=True)
        }
    
    def _get_project_status(self, start_date: str, end_date: str) -> str:
        """
        Determine project status based on dates
        
        Args:
            start_date (str): Project start date
            end_date (str): Project end date
            
        Returns:
            str: Project status
        """
        now = datetime.now()
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if now < start:
            return "Upcoming"
        elif start <= now <= end:
            return "Active"
        else:
            return "Completed"