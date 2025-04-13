import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
from itertools import product
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset: including the answer appended to the input text
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
        answer_text = item["answer"].strip()  # e.g. "Option 1" or "Option 2"
        # Build input that includes question, options, and answer (using [SEP] as delimiter)
        input_text = (
            f"Question: {question} [SEP] "
            f"Option 1: {option1} [SEP] "
            f"Option 2: {option2} [SEP] "
            f"Answer: {answer_text}"
        )
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt",
        )
        # Map the answer to binary label: 0 if "Option 1", else 1.
        label = 0 if answer_text.lower().startswith("option 1") else 1
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop function; returns validation accuracy and trained model.
def train_model(train_dataloader, dev_dataloader, learning_rate, epochs):
    # Initialize a fresh model each run.
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)
    model.train()
    for epoch in range(epochs):
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
    
    # Evaluate on validation set.
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

# Paths to the data files (adjust if needed)
train_file = "QQA/QQA_train.json"
dev_file   = "QQA/QQA_dev.json"
test_file  = "QQA/QQA_test.json"

# Load tokenizer and datasets.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# train_dataset = QQADataset(train_file, tokenizer, max_length=256)
# dev_dataset   = QQADataset(dev_file, tokenizer, max_length=256)
# test_dataset  = QQADataset(test_file, tokenizer, max_length=256)

# # Grid search space for hyperparameter tuning.
# learning_rates = [5e-5] # , 3e-5, 2e-5]
# batch_sizes = [8] #, 16]
# epochs_list = [2] #, 3]

# best_accuracy = 0
# best_config = None
# best_model = None

# print("Starting hyperparameter tuning...\n")
# for lr, bs, ep in product(learning_rates, batch_sizes, epochs_list):
#     print(f"Config: LR={lr}, Batch Size={bs}, Epochs={ep}")
#     train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
#     dev_loader   = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=bs)
    
#     val_accuracy, model = train_model(train_loader, dev_loader, lr, ep)
#     print(f"  --> Validation Accuracy: {val_accuracy:.4f}")
    
#     if val_accuracy > best_accuracy:
#         best_accuracy = val_accuracy
#         best_config = (lr, bs, ep)
#         best_model = model

# print(f"\nBest configuration: LR={best_config[0]}, Batch Size={best_config[1]}, Epochs={best_config[2]} with Validation Accuracy={best_accuracy:.4f}")

# # Save the best model and the tokenizer.
output_dir = "./best_qqa_model/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# best_model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"Best model saved to {output_dir}")

# # Test evaluation using the best batch size.
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

# print(f"\nTest Accuracy: {total_test_accuracy / test_steps:.4f}")

# ---------- Inference Code ----------

# Function to perform inference on a new example.
def infer(input_example, model, tokenizer, max_length=256):
    """
    input_example: A dictionary with keys: "question", "Option1", "Option2".
    model: The fine-tuned model.
    tokenizer: The corresponding tokenizer.
    """
    question = input_example["question"].strip()
    option1 = input_example["Option1"].strip()
    option2 = input_example["Option2"].strip()
    # Note: during inference we typically don't have a gold answer. Here, we just omit it
    input_text = f"Question: {question} [SEP] Option 1: {option1} [SEP] Option 2: {option2}"
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.detach().cpu().numpy()
    pred = np.argmax(logits, axis=1)[0]
    
    # Return the chosen option.
    return option1 if pred == 0 else option2

# Load the saved model and tokenizer for inference.
loaded_model = BertForSequenceClassification.from_pretrained(output_dir)
loaded_model.to(device)
loaded_tokenizer = BertTokenizer.from_pretrained(output_dir)

# Example inference on a new QQA example.
test_examples = [
    {
        "question": "A car with a mass of 1500 kg and a truck with a mass of 3000 kg both apply the same engine force. Which will accelerate faster?",
        "Option1": "The car",
        "Option2": "The truck"
    },
    {
        "question": "Two cyclists travel along a track at the same constant speed. Cyclist A rides for 30 minutes, while Cyclist B rides for 45 minutes. Who covers the greater distance?",
        "Option1": "Cyclist A",
        "Option2": "Cyclist B"
    },
    {
        "question": "A smartphone weighing 180 grams and a tablet weighing 400 grams fall from the same height with no air resistance. Which one will hit the ground first?",
        "Option1": "The smartphone",
        "Option2": "The tablet"
    },
    {
        "question": "A cyclist on a smooth road travels at 25 km/hr, while the same cyclist on a rough road travels at 15 km/hr. Which road has less resistance?",
        "Option1": "The smooth road",
        "Option2": "The rough road"
    },
    {
        "question": "A balloon filled with helium weighs 200 grams and a similar balloon filled with air weighs 250 grams. Assuming the lifting force is constant, which will accelerate upward faster?",
        "Option1": "The helium balloon",
        "Option2": "The air-filled balloon"
    },
    {
        "question": "A runner accelerates from rest at 4 m/s² while a jogger accelerates at 2 m/s². Which person reaches a higher speed in the same time interval?",
        "Option1": "The runner",
        "Option2": "The jogger"
    },
    {
        "question": "Two scooters travel side-by-side. One scooter is heavier and one is lighter. If both scooters are given the same push, which one will move faster initially?",
        "Option1": "The heavier scooter",
        "Option2": "The lighter scooter"
    },
    {
        "question": "A table is 80 cm long and a desk is 120 cm long. If you view them from the same distance, which one appears larger?",
        "Option1": "The table",
        "Option2": "The desk"
    },
    {
        "question": "A metal pipe conducts heat faster than a plastic pipe. If both are heated by the same source, which one reaches a higher temperature first?",
        "Option1": "The metal pipe",
        "Option2": "The plastic pipe"
    },
    {
        "question": "Two boats move at 20 km/hr. Boat A travels continuously for 1.5 hours and Boat B for 2 hours. Which boat covers more distance?",
        "Option1": "Boat A",
        "Option2": "Boat B"
    }
]

for example in test_examples:
    predicted_option = infer(example, loaded_model, loaded_tokenizer)
    print("\nInference Example:")
    print(f"Question: {example['question']}")
    print(f"Predicted Answer: {predicted_option}")
