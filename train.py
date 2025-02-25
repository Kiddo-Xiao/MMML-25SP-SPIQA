import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import MultimodalDataset
from model import MultimodalClassifier

# Use GPU in Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Access dataset
csv_path = "OverheatData/training_data.csv"
data_dir = "OverheatData"
dataset = MultimodalDataset(csv_path, data_dir)

# Data Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Data Loader
BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Initialize
num_classes = len(set(dataset.answer_to_label.values()))  # 计算总类别数
model = MultimodalClassifier(num_classes=num_classes).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Optimizer & Loss Function
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 10

# Train Loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss, correct_train, total_train = 0, 0, 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch["question"] = batch["question"].to(device)
        batch["image"] = batch["image"].to(device)
        batch["label"] = batch["label"].to(device)

        output_logits = model(batch["question"], batch["image"])
        loss = criterion(output_logits, batch["label"])
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(output_logits, dim=-1)
        correct_train += (preds == batch["label"]).sum().item()
        total_train += batch["label"].numel()

    train_loss = total_train_loss / len(train_loader)  
    train_acc = (correct_train / total_train) * 100  

    # Validation
    model.eval()
    total_val_loss, correct_val, total_val = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            batch["question"] = batch["question"].to(device)
            batch["image"] = batch["image"].to(device)
            batch["label"] = batch["label"].to(device)

            output_logits = model(batch["question"], batch["image"])
            loss = criterion(output_logits, batch["label"])

            total_val_loss += loss.item()
            preds = torch.argmax(output_logits, dim=-1)
            correct_val += (preds == batch["label"]).sum().item()
            total_val += batch["label"].numel()

    val_acc = (correct_val / total_val) * 100
    val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
