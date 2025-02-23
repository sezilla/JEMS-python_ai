import json
import torch
import datasets
import torch.nn as nn
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertPreTrainedModel,
    DistilBertModel,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
DATASET_PATH = "synthetic_project_dataset.json"
print("Loading dataset...")
with open(DATASET_PATH, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples.")

# Tokenizer
MODEL_NAME = "distilbert-base-uncased"
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# Encode labels using MultiLabelBinarizer
print("Encoding labels...")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform([item["allocated_teams"] for item in data])
num_labels = len(mlb.classes_)  # Number of possible labels
label2id = {str(label): i for i, label in enumerate(mlb.classes_)}
id2label = {i: str(label) for i, label in enumerate(mlb.classes_)}
print(f"Labels encoded. Number of labels: {num_labels}")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["project_text"],  # Use correct dataset key
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Convert dataset to Hugging Face format
dataset_dict = {
    "project_text": [f"{item['project_name']} {item['package_id']} {item['start']} {item['end']}" for item in data],
    "labels": labels.astype(np.float32).tolist()  # Ensure float conversion before mapping
}
dataset = datasets.Dataset.from_dict(dataset_dict)
print("Dataset converted.")

# Tokenize dataset
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["project_text"])
print("Tokenization completed.")

# Ensure labels are properly formatted
dataset = dataset.map(lambda x: {"labels": torch.tensor(x["labels"], dtype=torch.float32).numpy()})
print("Labels converted to float tensors.")

# Split dataset
print("Splitting dataset...")
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]
print("Dataset split completed.")

# Custom DistilBERT model for Multi-Label Classification
class DistilBertForMultiLabelClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for multi-label classification
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use CLS token

        loss = None
        if labels is not None:
            labels = labels.float()  # Ensure labels are float before loss computation
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# Load model
print("Loading model...")
model = DistilBertForMultiLabelClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
)
print("Model loaded.")

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)
print("Training arguments set.")

# Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print("Trainer initialized.")

# Train model
print("Starting training...")
trainer.train()
print("Training completed.")

# Save model
print("Saving model...")
model.save_pretrained("./distilbert_team_allocator")
tokenizer.save_pretrained("./distilbert_team_allocator")
print("Model training complete and saved.")
