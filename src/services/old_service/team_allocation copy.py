import json
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TeamAllocator")

# Constants and configuration
LOCAL_MODEL_PATH = "src/classifier/baseModel/JEMS_team_allocator"
HF_MODEL_PATH = "sezilla/team_allocator"  # Fallback to HuggingFace hosted model
HISTORY_DIR = "src/classifier/history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "team_allocation.json")
RAG_INDEX_PATH = os.path.join(LOCAL_MODEL_PATH, "embeddings/faiss_index.bin")
RAG_DOCS_PATH = os.path.join(LOCAL_MODEL_PATH, "embeddings/documents.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Create necessary directories
os.makedirs(HISTORY_DIR, exist_ok=True)

class TeamAllocatorRAG:
    """RAG component for the team allocator service"""
    
    def __init__(self, embedding_model_name: str):
        logger.info(f"🔍 Initializing RAG system with {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
    def load_index(self, index_path: str, docs_path: str) -> bool:
        """Load the FAISS index and documents from the given paths"""
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                logger.info(f"📥 Loading RAG index from {index_path}")
                self.index = faiss.read_index(index_path)
                
                logger.info(f"📥 Loading RAG documents from {docs_path}")
                with open(docs_path, "r") as f:
                    self.documents = json.load(f)
                    
                logger.info(f"✅ Loaded RAG index with {len(self.documents)} documents")
                return True
            except Exception as e:
                logger.error(f"❌ Error loading RAG index: {e}")
                return False
        else:
            logger.warning(f"⚠️ RAG index or documents file not found")
            return False
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve similar documents for a query"""
        if self.index is None:
            logger.warning("⚠️ RAG index not loaded - cannot retrieve documents")
            return []
            
        logger.info(f"🔍 Retrieving similar projects for: {query}")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "distance": float(distance)
                })
                
        logger.info(f"✅ Retrieved {len(results)} similar projects")
        return results

class EventTeamAllocator:
    def __init__(self):
        # Initialize team schedules from history
        self.team_schedules = {}
        self.project_history = self.load_project_history()
        self.update_team_schedules_from_history()
        
        # Initialize RAG system
        self.rag = TeamAllocatorRAG(EMBEDDING_MODEL)
        self.rag_enabled = self.rag.load_index(RAG_INDEX_PATH, RAG_DOCS_PATH)
        
        # Load the trained model
        logger.info("📥 Loading trained model...")
        try:
            # Try to load local model first
            if os.path.exists(LOCAL_MODEL_PATH):
                logger.info(f"📥 Loading model from local path: {LOCAL_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
                logger.info("✅ Local model loaded successfully")
            else:
                # Fall back to Hugging Face hosted model
                logger.info(f"📥 Local model not found, loading from Hugging Face: {HF_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)
                logger.info("✅ Hugging Face model loaded successfully")
                
            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise RuntimeError(f"Failed to load team allocator model: {e}")

    def load_project_history(self) -> List[Dict]:
        """Load project history from JSON file"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as file:
                    if os.path.getsize(HISTORY_FILE) == 0:
                        logger.warning("⚠️ History file is empty. Initializing with empty list.")
                        return []
                        
                    data = json.load(file)
                    logger.info(f"📂 Loaded project history with {len(data)} records")
                    return data
            except json.JSONDecodeError:
                logger.warning("⚠️ History file is corrupt. Resetting file.")
                return []
        
        logger.info("📂 No existing history file found. Creating a new one.")
        return []
    
    def save_project_history(self) -> None:
        """Save project history to JSON file"""
        try:
            with open(HISTORY_FILE, "w") as file:
                json.dump(self.project_history, file, indent=4)
            logger.info(f"✅ Project history saved with {len(self.project_history)} records")
        except Exception as e:
            logger.error(f"❌ Error saving project history: {e}")
    
    def update_team_schedules_from_history(self) -> None:
        """Update team schedules from project history"""
        self.team_schedules = {}
        
        for project in self.project_history:
            start_date = datetime.strptime(project["start"], "%Y-%m-%d")
            end_date = datetime.strptime(project["end"], "%Y-%m-%d")
            
            for team_id in project.get("allocated_teams", []):
                team_id = int(team_id) if isinstance(team_id, str) else team_id
                self.team_schedules.setdefault(team_id, []).append((start_date, end_date))
        
        # Count schedules per team
        schedule_counts = {team_id: len(schedules) for team_id, schedules in self.team_schedules.items()}
        logger.info(f"📊 Updated {len(self.team_schedules)} team schedules from history")
        
# Continue from where the code left off
        if schedule_counts:
            most_scheduled = max(schedule_counts.items(), key=lambda x: x[1])
            logger.info(f"📊 Most scheduled team: Team {most_scheduled[0]} with {most_scheduled[1]} projects")
    
    def is_team_available(self, team_id: int, start_date: datetime, end_date: datetime) -> bool:
        """Check if a team is available during the specified date range"""
        if team_id not in self.team_schedules:
            return True  # Team has no schedules, so it's available
            
        for scheduled_start, scheduled_end in self.team_schedules[team_id]:
            # Check for overlap
            if not (end_date < scheduled_start or start_date > scheduled_end):
                return False
                
        return True
    
    def get_team_load(self, team_id: int) -> int:
        """Get the number of projects assigned to a team"""
        return len(self.team_schedules.get(team_id, []))
    
    def predict_teams(self, project_name: str, package_id: int, start_date_str: str, end_date_str: str) -> List[int]:
        """
        Predict the best teams for a project using the trained model
        """
        logger.info(f"🔮 Predicting teams for project: {project_name}, package: {package_id}")
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Create feature text
        feature_text = f"Project: {project_name} Package: {package_id} Start: {start_date_str} End: {end_date_str}"
        
        # Gather RAG context if available
        rag_context = ""
        if self.rag_enabled:
            similar_projects = self.rag.retrieve(feature_text, k=3)
            
            if similar_projects:
                rag_context = " Similar projects: "
                for i, result in enumerate(similar_projects[:3], 1):
                    doc = result["document"]
                    distance = result["distance"]
                    # Only include if it's reasonably similar (lower distance is better in FAISS)
                    if distance < 50:  # Threshold for similarity
                        rag_context += f"[{i}]{doc['project_name']}(teams:{','.join(map(str, doc['allocated_teams']))}) "
        
        # Augment feature text with RAG context
        augmented_text = feature_text + rag_context
        logger.info(f"📝 Using augmented text for prediction: {augmented_text}")
        
        # Tokenize the input
        inputs = self.tokenizer(
            augmented_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Get teams with probability > 0.5
        team_ids = []
        threshold = 0.5
        
        # Get team IDs from model prediction
        for i, prob in enumerate(probs[0]):
            if prob > threshold:
                team_id = i + 1  # Adjust index to team_id (1-based)
                team_ids.append(team_id)
        
        # If no teams predicted with high confidence, take top 3
        if not team_ids:
            logger.warning("⚠️ No teams predicted with high confidence. Using top 3.")
            _, indices = torch.topk(probs[0], k=min(3, len(probs[0])))
            team_ids = [int(idx) + 1 for idx in indices.tolist()]  # Convert to 1-based team IDs

        logger.info(f"✅ Model predicted teams: {team_ids}")
        
        # Filter by availability and optimize team allocation
        final_teams = self.optimize_team_allocation(team_ids, start_date, end_date)
        
        return final_teams
    
    def optimize_team_allocation(self, predicted_teams: List[int], start_date: datetime, end_date: datetime) -> List[int]:
        """Optimize team allocation based on availability and workload"""
        logger.info(f"⚙️ Optimizing team allocation from {len(predicted_teams)} predicted teams")
        
        # Define team structure (department mappings)
        department_teams = {
            1: [1, 2, 3, 4, 5, 6],          # Catering
            2: [7, 8, 9, 10, 11, 12],       # Hair and Makeup
            3: [13, 14, 15, 16, 17, 18],    # Photo and Video
            4: [19, 20, 21, 22, 23, 24],    # Designing
            5: [25, 26, 27, 28, 29, 30],    # Entertainment
            6: [31, 32, 33, 34, 35, 36],    # Coordination
        }
        
        # Reverse mapping from team to department
        team_to_department = {}
        for dept_id, teams in department_teams.items():
            for team_id in teams:
                team_to_department[team_id] = dept_id
        
        # Track departments that have been allocated
        allocated_departments = set()
        final_teams = []
        
        # First pass: Check if predicted teams are available
        for team_id in predicted_teams:
            dept_id = team_to_department.get(team_id)
            
            # Skip if we already allocated a team from this department
            if dept_id in allocated_departments:
                continue
                
            # Check if team is available
            if self.is_team_available(team_id, start_date, end_date):
                final_teams.append(team_id)
                allocated_departments.add(dept_id)
                logger.info(f"✅ Team {team_id} from department {dept_id} is available and selected")
            else:
                logger.info(f"⚠️ Team {team_id} is not available, looking for alternatives")
        
        # Second pass: For departments that don't have an allocated team yet,
        # find the best available team from that department
        for team_id in predicted_teams:
            dept_id = team_to_department.get(team_id)
            
            # Skip if we already allocated a team from this department
            if dept_id in allocated_departments:
                continue
                
            # Find best available team from this department
            best_team = None
            lowest_load = float('inf')
            
            for alt_team_id in department_teams.get(dept_id, []):
                if self.is_team_available(alt_team_id, start_date, end_date):
                    team_load = self.get_team_load(alt_team_id)
                    if team_load < lowest_load:
                        lowest_load = team_load
                        best_team = alt_team_id
            
            if best_team is not None:
                final_teams.append(best_team)
                allocated_departments.add(dept_id)
                logger.info(f"✅ Found alternative: Team {best_team} from department {dept_id}")
            else:
                # If no team is available, pick the team with least workload
                alt_teams = [(t, self.get_team_load(t)) for t in department_teams.get(dept_id, [])]
                if alt_teams:
                    best_team = min(alt_teams, key=lambda x: x[1])[0]
                    final_teams.append(best_team)
                    allocated_departments.add(dept_id)
                    logger.warning(f"⚠️ No available teams in department {dept_id}, selected least busy team {best_team}")
        
        logger.info(f"✅ Final optimized team allocation: {final_teams}")
        return final_teams
    
    def allocate_teams(self, project_name: str, package_id: int, start_date: str, end_date: str) -> Dict:
        """
        Allocate teams for a new project and save to history
        """
        logger.info(f"🚀 Allocating teams for new project: {project_name}")
        
        # Predict the best teams for this project
        allocated_teams = self.predict_teams(project_name, package_id, start_date, end_date)
        
        # Create project record
        project = {
            "project_name": project_name,
            "package_id": package_id,
            "start": start_date,
            "end": end_date,
            "allocated_teams": allocated_teams,
            "allocation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to history
        self.project_history.append(project)
        self.save_project_history()
        
        # Update team schedules
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        for team_id in allocated_teams:
            self.team_schedules.setdefault(team_id, []).append((start_dt, end_dt))
        
        logger.info(f"✅ Successfully allocated {len(allocated_teams)} teams to project {project_name}")
        return project

def main():
    logger.info("🚀 Starting Team Allocator Service")
    
    try:
        # Initialize the team allocator
        allocator = EventTeamAllocator()
        
        # Demo: Allocate teams for a new project
        project_name = "Aurora Gala"
        package_id = 3  # Premium package
        start_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d")
        
        logger.info(f"🎯 Demo: Allocating teams for {project_name}, Package: {package_id}")
        logger.info(f"📅 Project dates: {start_date} to {end_date}")
        
        result = allocator.allocate_teams(project_name, package_id, start_date, end_date)
        
        logger.info("✨ Demo completed successfully")
        logger.info(f"📋 Teams allocated: {result['allocated_teams']}")
        logger.info(f"💾 Project saved to history file: {HISTORY_FILE}")
        
    except Exception as e:
        logger.error(f"❌ Error in Team Allocator Service: {e}", exc_info=True)

if __name__ == "__main__":
    main()