import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for QQA data (using the primary question and options)
class QQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"].strip()
        option1 = item["Option1"].strip()
        option2 = item["Option2"].strip()
        # Create a single input string (you can use "[SEP]" as a delimiter)
        input_text = f"Question: {question} [SEP] Option 1: {option1} [SEP] Option 2: {option2}"
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Map the answer to a binary label: 0 for "Option 1", 1 for "Option 2"
        ans = item["answer"].strip().lower()
        label = 0 if ans.startswith("option 1") else 1
        
        # Squeeze to remove the batch dim and return dictionary
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Paths to files – adjust if needed.
train_file = "QQA/QQA_train.json"
dev_file = "QQA/QQA_dev.json"
test_file = "QQA/QQA_test.json"

# Initialize tokenizer and create datasets
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = QQADataset(train_file, tokenizer, max_length=256)
dev_dataset   = QQADataset(dev_file, tokenizer, max_length=256)
test_dataset  = QQADataset(test_file, tokenizer, max_length=256)

# Create DataLoaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
dev_dataloader   = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size)
test_dataloader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Initialize model (binary classification -> num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Define optimizer and scheduler
epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Loss function (model already includes classification head and its loss when given labels)
loss_fn = nn.CrossEntropyLoss()

def flat_accuracy(preds, labels):
    # Calculate the accuracy
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        
        model.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        
        loss.backward()
        # Clip the norm of gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 50 == 0:
            print(f"  Batch {step+1}/{len(train_dataloader)}  Loss: {loss.item():.4f}")
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in dev_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        nb_eval_steps += 1
    
    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    avg_val_loss = total_eval_loss / nb_eval_steps
    print(f"Validation Loss: {avg_val_loss:.4f}  Accuracy: {avg_val_accuracy:.4f}")

# Test inference: compute accuracy on test set
model.eval()
total_test_accuracy = 0
nb_test_steps = 0

for batch in test_dataloader:
    b_input_ids = batch["input_ids"].to(device)
    b_attention_mask = batch["attention_mask"].to(device)
    b_labels = batch["labels"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to("cpu").numpy()
    total_test_accuracy += flat_accuracy(logits, label_ids)
    nb_test_steps += 1

avg_test_accuracy = total_test_accuracy / nb_test_steps
print(f"\nTest Accuracy: {avg_test_accuracy:.4f}")
