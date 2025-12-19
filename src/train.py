from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
from torch.utils.data import DataLoader,random_split 

from tqdm import tqdm
import os

import pandas as pd
import numpy as np

from dataset import TextDataset
from metrics import compute_metrics



# Root project directory (assumes train.py is in src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")


# Path to the CSV relative to root
data_path = os.path.join(BASE_DIR, "data", "Grammer Correction.csv")
df = pd.read_csv(data_path)
df.drop(columns={'Serial Number','Error Type'},inplace=True)

texts = df['Ungrammatical Statement'].tolist()
labels = df['Standard English'].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction").to(device)
tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")


dataset = TextDataset(texts, labels, tokenizer, max_length=128)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)


model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


num_epochs = 25
patience = 4
best_rougeL = 0
counter = 0

for epoch in range(num_epochs):

    # Train 
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in loop:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=128
            )

            preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            labels = tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )

            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_metrics(all_preds, all_labels)

    print("\n================ RESULTS =================")
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"Normalized Edit Distance: {metrics['norm_edit_distance']:.4f}")

    # Early stopping 
    current_rougeL = metrics["rougeL"]

    if current_rougeL > best_rougeL:
        best_rougeL = current_rougeL
        counter = 0

        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        print(f"model saved (ROUGE-L = {best_rougeL:.4f})")
    else:
        counter += 1
        print(f" No improvement ({counter}/{patience})")

    if counter >= patience:
        print("Early stopping triggered")
        break
