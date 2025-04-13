import json
import torch
import torch.nn as nn
import numpy as np
from itertools import product
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup , BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualBERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, num_labels=2):
        super(DualBERTClassifier, self).__init__()
        self.bert_question = BertModel.from_pretrained(model_name)
        self.bert_sci = BertModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids_q, attention_mask_q, input_ids_sci, attention_mask_sci):
        # Get [CLS] token output from both BERTs
        out_q = self.bert_question(input_ids=input_ids_q, attention_mask=attention_mask_q).last_hidden_state[:, 0]
        out_sci = self.bert_sci(input_ids=input_ids_sci, attention_mask=attention_mask_sci).last_hidden_state[:, 0]

        # Concatenate both [CLS] outputs
        combined = torch.cat([out_q, out_sci], dim=1)

        # Classification head
        logits = self.classifier(combined)
        return logits


# --- Dataset ---
class QQADualEncoderDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        with open(file_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item["question"].strip()
        sci = item["question_sci_10E"].strip()
        option1 = item["Option1"].strip()
        option2 = item["Option2"].strip()
        label = 0 if item["answer"].lower().startswith("option 1") else 1

        input_q = self.tokenizer.encode_plus(
            f"{q} [SEP] Option 1: {option1} [SEP] Option 2: {option2}",
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        input_sci = self.tokenizer.encode_plus(
            f"{sci} [SEP] Option 1: {option1} [SEP] Option 2: {option2}",
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids_q": input_q["input_ids"].squeeze(0),
            "attention_mask_q": input_q["attention_mask"].squeeze(0),
            "input_ids_sci": input_sci["input_ids"].squeeze(0),
            "attention_mask_sci": input_sci["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --- Accuracy function ---
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# --- Training ---
def train_model(train_dataloader, dev_dataloader, learning_rate, epochs):
    model = DualBERTClassifier().to(device)  # <-- Use Dual Encoder here

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
            input_ids_q = batch["input_ids_q"].to(device)
            attn_q = batch["attention_mask_q"].to(device)
            input_ids_sci = batch["input_ids_sci"].to(device)
            attn_sci = batch["attention_mask_sci"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            logits = model(input_ids_q, attn_q, input_ids_sci, attn_sci)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    # Evaluation
    model.eval()
    total_accuracy = 0
    steps = 0
    for batch in dev_dataloader:
        input_ids_q = batch["input_ids_q"].to(device)
        attn_q = batch["attention_mask_q"].to(device)
        input_ids_sci = batch["input_ids_sci"].to(device)
        attn_sci = batch["attention_mask_sci"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = model(input_ids_q, attn_q, input_ids_sci, attn_sci)

        preds = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        total_accuracy += flat_accuracy(preds, labels)
        steps += 1

    return total_accuracy / steps, model

# --- Load Data ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = QQADualEncoderDataset("QQA/QQA_train.json", tokenizer)
dev_dataset = QQADualEncoderDataset("QQA/QQA_dev.json", tokenizer)
# test_dataset = QQAConcatDataset("QQA/QQA_test.json", tokenizer)

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

class QQADualEncoderTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = item["question"].strip()
        sci = item["question_sci_10E"].strip()
        option1 = item["Option1"].strip()
        option2 = item["Option2"].strip()

        input_q = self.tokenizer.encode_plus(
            f"{q} [SEP] Option 1: {option1} [SEP] Option 2: {option2}",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_sci = self.tokenizer.encode_plus(
            f"{sci} [SEP] Option 1: {option1} [SEP] Option 2: {option2}",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids_q": input_q["input_ids"].squeeze(0),
            "attention_mask_q": input_q["attention_mask"].squeeze(0),
            "input_ids_sci": input_sci["input_ids"].squeeze(0),
            "attention_mask_sci": input_sci["attention_mask"].squeeze(0)
        }

with open("QQA/QQA_test.json", "r") as f:
    raw_test_data = json.load(f)

# Load test JSONtest_dataset = QQATestOnlyDataset(raw_test_data, tokenizer)
test_dataset = QQADualEncoderTestDataset(raw_test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=best_config[1], sampler=SequentialSampler(test_dataset))

# Predict
predictions = []
best_model.eval()
for batch in test_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        logits = best_model(input_ids=input_ids, attention_mask=attention_mask)
    preds = torch.argmax(logits.logits, dim=1).cpu().tolist()
    predictions.extend(preds)

# Evaluate accuracy if 'answer' exists
correct = 0
total = 0

for i, pred in enumerate(predictions):
    gold = raw_test_data[i].get("answer", "").strip().lower()
    gold_label = 0 if gold.startswith("option 1") else 1
    raw_test_data[i]["predicted_answer"] = "Option 1" if pred == 0 else "Option 2"
    raw_test_data[i]["predicted_label"] = pred
    raw_test_data[i]["gold_label"] = gold_label
    if gold:
        correct += int(pred == gold_label)
        total += 1

# Compute and report accuracy
if total > 0:
    test_accuracy = correct / total
    print(f"\n✅ Test Accuracy (with ground truth): {test_accuracy:.4f}")
else:
    print("⚠️ Test set does not contain 'answer' field. Skipping accuracy computation.")

# Save output
with open("QQA/QQA_test_predictions.json", "w") as f:
    json.dump(raw_test_data, f, indent=2)
print("✅ Predictions saved to QQA/QQA_Concat_test_predictions.json")
