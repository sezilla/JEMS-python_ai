import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import faiss
import shutil

class TeamAllocationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TeamAllocatorTrainer:
    def __init__(self):
        self.base_model_path = "src/classifier/baseModel/JEMS_team_allocator/"
        self.dataset_path = "src/datasets/team_allocator_dataset.json"
        self.history_path = "src/classifier/history/team_allocation.json"
        
        # Ensure directories exist
        os.makedirs(self.base_model_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        
        # Model parameters
        self.embedding_size = 128
        self.hidden_size = 256
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 20
        
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
        
        # RAG components
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.rag_data = []
    
    def prepare_data(self):
        """Load and prepare data for training"""
        print("Loading dataset...")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            raise ValueError("Dataset is empty. Run the dataset generator first.")
        
        print(f"Loaded {len(data)} projects from dataset")
        
        # Load history if exists
        history_data = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    history_data = json.load(f)
                print(f"Loaded {len(history_data)} historical records")
            except:
                print("Could not load history or history file is empty")
        
        # Combine dataset with history for training
        combined_data = data + history_data
        
        # Process dates
        for item in combined_data:
            if "allocation_date" in item:
                item.pop("allocation_date", None)
            if "message" in item:
                item.pop("message", None)
        
        # Prepare features and labels by department
        self.department_features = {dept_id: [] for dept_id in self.department_teams}
        self.department_labels = {dept_id: [] for dept_id in self.department_teams}
        
        for item in combined_data:
            start_date = datetime.strptime(item["start"], "%Y-%m-%d")
            end_date = datetime.strptime(item["end"], "%Y-%m-%d")
            project_duration = (end_date - start_date).days
            
            # Extract features
            features = [
                item["project_id"],
                item["package_id"],
                start_date.year,
                start_date.month,
                start_date.day,
                end_date.year,
                end_date.month,
                end_date.day,
                project_duration
            ]
            
            # Extract allocated teams and map to departments
            allocated_teams = item["allocated_teams"]
            
            # Map departments to their allocated teams
            dept_team_map = {}
            for team_id in allocated_teams:
                for dept_id, teams in self.department_teams.items():
                    if team_id in teams:
                        dept_team_map[dept_id] = team_id
                        break
            
            # Prepare data for each department in the package
            package_id = item["package_id"]
            for dept_id in self.package_departments[package_id]:
                dept_features = features.copy()
                
                # Department-specific feature
                dept_features.append(dept_id)
                
                # Convert feature to tensor
                if dept_id in dept_team_map:
                    self.department_features[dept_id].append(dept_features)
                    self.department_labels[dept_id].append(dept_team_map[dept_id])
        
        print("Data preparation complete")
        
        # Set up RAG index with prepared data
        self._setup_rag_index(combined_data)
        
        return True
    
    def _setup_rag_index(self, data):
        """Setup FAISS index for RAG"""
        print("Setting up RAG index...")
        
        # Prepare data for RAG
        self.rag_data = []
        texts = []
        
        for item in data:
            # Create text representation
            text = f"Project {item['project_id']} with package {item['package_id']} from {item['start']} to {item['end']} teams: {item['allocated_teams']}"
            texts.append(text)
            self.rag_data.append(item)
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"RAG index created with {len(texts)} documents")
    
    def _retrieve_similar_projects(self, query, k=5):
        """Retrieve similar projects for RAG enhancement"""
        query_embedding = self.sentence_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        retrieved = [self.rag_data[idx] for idx in indices[0]]
        return retrieved
    
    def train(self):
        """Train the team allocation model for each department"""
        if not self.prepare_data():
            return False
        
        print("Starting model training...")
        
        # Train a model for each department
        for dept_id in self.department_teams:
            if not self.department_features[dept_id]:
                print(f"No training data for department {dept_id}, skipping...")
                continue
                
            print(f"\nTraining model for Department {dept_id}")
            
            # Convert features and labels to PyTorch tensors
            X = torch.tensor(self.department_features[dept_id], dtype=torch.float32)
            
            # One-hot encode labels (team IDs)
            teams = self.department_teams[dept_id]
            team_to_idx = {team: idx for idx, team in enumerate(teams)}
            y_idx = [team_to_idx[team] for team in self.department_labels[dept_id]]
            y = torch.tensor(y_idx, dtype=torch.long)
            
            # Create dataset and dataloader
            dataset = TeamAllocationDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Define model
            model = nn.Sequential(
                nn.Linear(11, self.hidden_size),  # Change from 10 → 11
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size, len(teams))
            )
            
            # Load existing model if it exists
            model_path = os.path.join(self.base_model_path, f"department_{dept_id}_model.pt")
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path))
                    print(f"Loaded existing model for Department {dept_id}")
                except:
                    print(f"Couldn't load existing model. Training new model for Department {dept_id}")
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Training loop
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_features, batch_labels in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                # Print epoch statistics
                epoch_loss = running_loss / len(dataloader)
                epoch_acc = 100 * correct / total
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
            # Save the model
            torch.save(model.state_dict(), model_path)
            print(f"Model for Department {dept_id} saved to {model_path}")
            
            # Save model metadata
            metadata = {
                "teams": teams,
                "features": ["project_id", "package_id", "start_year", "start_month", "start_day", 
                             "end_year", "end_month", "end_day", "duration", "department_id"],
                "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.base_model_path, f"department_{dept_id}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)
        
        # Save RAG index
        try:
            faiss.write_index(self.index, os.path.join(self.base_model_path, "rag_index.faiss"))
            with open(os.path.join(self.base_model_path, "rag_data.json"), 'w') as f:
                json.dump(self.rag_data, f, indent=4)
            print("RAG components saved")
        except Exception as e:
            print(f"Error saving RAG components: {e}")
        
        print("\nTraining complete for all departments")
        return True

if __name__ == "__main__":
    trainer = TeamAllocatorTrainer()
    trainer.train()