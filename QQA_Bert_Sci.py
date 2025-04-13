import json
import torch
import numpy as np
from itertools import product
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
class QQASciDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question_sci_10E"].strip()
        option1 = item["Option1"].strip()
        option2 = item["Option2"].strip()
        input_text = f"Question: {question} [SEP] Option 1: {option1} [SEP] Option 2: {option2}"
        encoding = self.tokenizer.encode_plus(
            input_text, add_special_tokens=True, max_length=self.max_length,
            truncation=True, padding='max_length', return_attention_mask=True, return_tensors="pt"
        )
        label = 0 if item["answer"].strip().lower().startswith("option 1") else 1
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --- Accuracy function ---
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# --- Training ---
def train_model(train_dataloader, dev_dataloader, learning_rate, epochs):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            b_input_ids = batch["input_ids"].to(device)
            b_attention_mask = batch["attention_mask"].to(device)
            b_labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    # Evaluation
    model.eval()
    total_accuracy = 0
    steps = 0
    for batch in dev_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits.detach().cpu().numpy()
        labels = b_labels.cpu().numpy()
        total_accuracy += flat_accuracy(logits, labels)
        steps += 1

    return total_accuracy / steps, model

# --- Load Data ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = QQASciDataset("QQA/QQA_train.json", tokenizer)
dev_dataset = QQASciDataset("QQA/QQA_dev.json", tokenizer)
# test_dataset = QQADataset("QQA/QQA_test.json", tokenizer)

# --- Hyperparameter Search ---
learning_rates = [5e-5, 3e-5, 2e-5]
batch_sizes = [8, 16]
epochs_list = [2, 3]

best_accuracy = 0
best_config = None
best_model = None

print("Starting hyperparameter tuning...\n")
for lr, bs, ep in product(learning_rates, batch_sizes, epochs_list):
    print(f"Trying config: LR={lr}, BS={bs}, Epochs={ep}")
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
    dev_loader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=bs)

    accuracy, model = train_model(train_loader, dev_loader, lr, ep)
    print(f"  --> Validation Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_config = (lr, bs, ep)
        best_model = model

print(f"\nBest config: LR={best_config[0]}, BS={best_config[1]}, Epochs={best_config[2]} with Accuracy={best_accuracy:.4f}")

# --- Save Best Model ---
save_path = "QQA_Sci/best_model_sci"
best_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n✅ Best model saved to {save_path}")

# # --- Evaluate on test set ---
# test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=best_config[1])
# best_model.eval()
# total_test_accuracy = 0
# test_steps = 0

# for batch in test_loader:
#     b_input_ids = batch["input_ids"].to(device)
#     b_attention_mask = batch["attention_mask"].to(device)
#     b_labels = batch["labels"].to(device)
#     with torch.no_grad():
#         outputs = best_model(input_ids=b_input_ids, attention_mask=b_attention_mask)
#     logits = outputs.logits.detach().cpu().numpy()
#     labels = b_labels.cpu().numpy()
#     total_test_accuracy += flat_accuracy(logits, labels)
#     test_steps += 1

# print(f"\nTest Accuracy using best config: {total_test_accuracy / test_steps:.4f}")

# --- Inference on test set and save predictions ---
print("\nRunning inference and saving predictions...")

class QQATestOnlyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"].strip()
        option1 = item["Option1"].strip()
        option2 = item["Option2"].strip()
        input_text = f"Question: {question} [SEP] Option 1: {option1} [SEP] Option 2: {option2}"
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# Load test JSON
with open("QQA/QQA_test.json", "r") as f:
    raw_test_data = json.load(f)

# Create inference dataset and loader
test_infer_dataset = QQATestOnlyDataset(raw_test_data, tokenizer)
test_infer_loader = DataLoader(test_infer_dataset, batch_size=best_config[1], sampler=SequentialSampler(test_infer_dataset))

# Run inference
predictions = []
best_model.eval()
for batch in test_infer_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
    preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    predictions.extend(preds)

# Add predictions to test data
for i, pred in enumerate(predictions):
    raw_test_data[i]["predicted_answer"] = "Option 1" if pred == 0 else "Option 2"

# Save predictions
with open("QQA_sci_test_predictions.json", "w") as f:
    json.dump(raw_test_data, f, indent=2)

print("✅ Predictions saved to QQA/QQA_test_predictions.json")
