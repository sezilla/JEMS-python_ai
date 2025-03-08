import json
import torch
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import shutil

# Configuration
DATASET_PATH = "src/datasets/team_allocator_dataset.json"
BASE_MODEL_NAME = "distilroberta-base"  # Efficient model with good performance for prediction tasks
OUTPUT_DIR = "src/classifier/baseModel/JEMS_team_allocator"
HISTORY_DIR = "src/classifier/history"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective embedding model for RAG
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
EVAL_STEPS = 100
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "embeddings"), exist_ok=True)

print(f"🚀 Starting Team Allocator model training process")
print(f"📋 Using base model: {BASE_MODEL_NAME}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📊 Dataset path: {DATASET_PATH}")

# Load dataset
print("📥 Loading dataset...")
with open(DATASET_PATH, "r") as f:
    data = json.load(f)
print(f"✅ Loaded {len(data)} samples from dataset")

# RAG implementation
class RAGSystem:
    def __init__(self, embedding_model_name):
        print(f"🔍 Initializing RAG system with {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
    def add_documents(self, documents):
        """Add documents to the RAG system"""
        self.documents = documents
        
        # Create text representations for each document
        texts = [f"{doc['project_name']} Package: {doc['package_id']} From: {doc['start']} To: {doc['end']}" 
                for doc in documents]
        
        print(f"🔢 Computing embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, 
                                                 convert_to_numpy=True)
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"✅ Built FAISS index with {len(embeddings)} embeddings")
        
    def save_index(self, path):
        """Save the FAISS index and documents"""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "embeddings/faiss_index.bin"))
            with open(os.path.join(path, "embeddings/documents.json"), "w") as f:
                json.dump(self.documents, f)
            print(f"✅ Saved RAG index and documents to {path}")
        
    def load_index(self, path):
        """Load the FAISS index and documents"""
        index_path = os.path.join(path, "embeddings/faiss_index.bin")
        docs_path = os.path.join(path, "embeddings/documents.json")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "r") as f:
                self.documents = json.load(f)
            print(f"✅ Loaded RAG index with {len(self.documents)} documents")
            return True
        return False
        
    def retrieve(self, query, k=5):
        """Retrieve similar documents for a query"""
        if self.index is None:
            return []
            
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "document": doc,
                    "distance": float(distance)
                })
        return results
        
    def augment_samples(self, samples, k=3):
        """Augment training samples with retrieved documents"""
        augmented_samples = []
        
        for sample in tqdm(samples, desc="Augmenting samples"):
            query = f"{sample['project_name']} Package: {sample['package_id']} From: {sample['start']} To: {sample['end']}"
            similar_docs = self.retrieve(query, k)
            
            # Include original sample
            augmented_samples.append(sample)
            
            # Include information from similar documents 
            for doc_info in similar_docs:
                doc = doc_info["document"]
                # Skip if it's the same document
                if doc["project_name"] == sample["project_name"]:
                    continue
                    
                # Create an augmented sample with retrieved context
                augmented_sample = {
                    "project_name": f"{sample['project_name']} (Similar to: {doc['project_name']})",
                    "package_id": sample["package_id"],
                    "start": sample["start"],
                    "end": sample["end"],
                    "allocated_teams": sample["allocated_teams"]
                }
                augmented_samples.append(augmented_sample)
                
        print(f"✅ Augmented dataset from {len(samples)} to {len(augmented_samples)} samples")
        return augmented_samples

# Initialize RAG system
rag_system = RAGSystem(EMBEDDING_MODEL)

# Try to load existing index or build a new one
if not rag_system.load_index(OUTPUT_DIR):
    print("📋 Building new RAG index from dataset")
    rag_system.add_documents(data)
    rag_system.save_index(OUTPUT_DIR)
    
# Augment dataset with RAG
data = rag_system.augment_samples(data)

# Process the data
print("🔄 Processing dataset for model training...")

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Encode labels using MultiLabelBinarizer
print("🏷️ Encoding team labels...")
mlb = MultiLabelBinarizer()

# Extract all team IDs from the dataset
all_team_ids = set()
for item in data:
    all_team_ids.update(item["allocated_teams"])
all_team_ids = list(all_team_ids)

# Ensure the MultiLabelBinarizer knows about all possible teams
mlb.fit([all_team_ids])

# Transform the allocated teams
labels = mlb.transform([item["allocated_teams"] for item in data])
num_labels = len(mlb.classes_)
print(f"✅ Encoded labels. Total number of possible teams: {num_labels}")

# Create feature texts that include project details
feature_texts = []
for item in data:
    text = f"Project: {item['project_name']} Package: {item['package_id']} Start: {item['start']} End: {item['end']}"
    feature_texts.append(text)

# Create dataset dictionary and convert to Hugging Face dataset
dataset_dict = {
    "text": feature_texts,
    "labels": labels.tolist()
}
hf_dataset = Dataset.from_dict(dataset_dict)

# Split dataset using pandas and convert back to Hugging Face datasets
df = pd.DataFrame({
    "text": feature_texts,
    "labels": labels.tolist()
})
train_val_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(0.9 * len(df))])
train_df, val_df = np.split(train_val_df.sample(frac=1, random_state=42), [int(0.85 * len(train_val_df))])
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"📊 Dataset splits: Training: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Data collator for efficient batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert labels to float via batched mapping
def convert_labels(batch):
    batch["labels"] = torch.tensor(batch["labels"]).float()
    return batch

tokenized_train_dataset = tokenized_train_dataset.map(convert_labels, batched=True)
tokenized_val_dataset = tokenized_val_dataset.map(convert_labels, batched=True)
tokenized_test_dataset = tokenized_test_dataset.map(convert_labels, batched=True)

# Custom compute_loss to ensure labels are floats
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.get("labels")
    if labels is not None and labels.dtype != torch.float:
        inputs["labels"] = labels.float()
    outputs = model(**inputs)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss

# Check for an existing model to continue training
if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    try:
        print("🔄 Found existing model. Loading for continued training...")
        model = AutoModelForSequenceClassification.from_pretrained(
            OUTPUT_DIR,
            problem_type="multi_label_classification",
            num_labels=num_labels
        )
        print("✅ Successfully loaded existing model for continued training")
        
        # Back up the current model before updating
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
        print(f"📦 Creating backup of current model to {backup_dir}")
        shutil.copytree(OUTPUT_DIR, backup_dir, ignore=shutil.ignore_patterns("*checkpoint*", "runs", "*.bin"))
        
    except Exception as e:
        print(f"⚠️ Error loading existing model: {e}")
        print("🔄 Initializing new model instead")
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            problem_type="multi_label_classification",
            num_labels=num_labels
        )
else:
    print("🔄 Initializing new model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        problem_type="multi_label_classification",
        num_labels=num_labels
    )

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=WARMUP_STEPS,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none"
)

# Metrics calculation for multi-label classification
def compute_metrics(pred):
    predictions = (pred.predictions > 0).astype(np.int32)
    labels = pred.label_ids.astype(np.int32)
    accuracy = np.mean((predictions == labels).astype(np.float32))
    
    precision_sum, recall_sum, f1_sum, count = 0, 0, 0, 0
    for i in range(predictions.shape[1]):
        pred_i = predictions[:, i]
        label_i = labels[:, i]
        true_positives = np.sum((pred_i == 1) & (label_i == 1))
        false_positives = np.sum((pred_i == 1) & (label_i == 0))
        false_negatives = np.sum((pred_i == 0) & (label_i == 1))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        count += 1
        
    macro_precision = precision_sum / count if count > 0 else 0
    macro_recall = recall_sum / count if count > 0 else 0
    macro_f1 = f1_sum / count if count > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if labels is not None and labels.dtype != torch.float:
            inputs["labels"] = labels.float()  # Ensure labels are float for BCEWithLogitsLoss
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


# Train the model
print("🏋️ Starting model training...")
train_result = trainer.train()
print(f"✅ Training completed in {train_result.metrics['train_runtime']:.2f} seconds")

# Evaluate the model
print("📊 Evaluating model on test dataset...")
eval_results = trainer.evaluate(tokenized_test_dataset)
print(f"📊 Test results: {eval_results}")

# Save the final model and tokenizer
print("💾 Saving the final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save model card with performance metrics
model_card = f"""
# JEMS Team Allocator Model

This model predicts team allocations for event projects based on project details, 
package requirements, and team availability.

## Model Information
- Base model: {BASE_MODEL_NAME}
- Number of labels: {num_labels}
- Training date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Performance Metrics
- Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}
- Precision: {eval_results.get('eval_precision', 'N/A'):.4f}
- Recall: {eval_results.get('eval_recall', 'N/A'):.4f}
- F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}

## Usage
This model is designed to predict optimal team allocations for event projects
while considering team availability and scheduling constraints.
"""

with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(model_card)

print("✅ Training pipeline completed successfully")
print(f"📁 Model saved to {OUTPUT_DIR}")
