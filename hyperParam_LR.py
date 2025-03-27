import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt

from dataset import MultimodalDataset
from model import MultimodalClassifier

'''
Learning Rate Hyperparameter Sweep
'''
def run_experiment(learning_rate):
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
    num_classes = len(set(dataset.answer_to_label.values()))
    model = MultimodalClassifier(num_classes=num_classes).to(device)

    # Optimizer & Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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
        train_losses.append(train_loss)
        train_accs.append(train_acc)

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
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# Learning rate values to test
learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]

# Store results
results = {}

# Run experiments
for lr in learning_rates:
    print(f"\nRunning experiment with learning rate: {lr}")
    results[lr] = run_experiment(lr)

# Plotting function
def plot_learning_rate_comparison(results):
    plt.figure(figsize=(15, 10))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    for lr, res in results.items():
        plt.plot(res['train_losses'], label=f'LR={lr} (Train)')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation Loss
    plt.subplot(2, 2, 2)
    for lr, res in results.items():
        plt.plot(res['val_losses'], label=f'LR={lr} (Validation)')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Training Accuracy
    plt.subplot(2, 2, 3)
    for lr, res in results.items():
        plt.plot(res['train_accs'], label=f'LR={lr} (Train)')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Validation Accuracy
    plt.subplot(2, 2, 4)
    for lr, res in results.items():
        plt.plot(res['val_accs'], label=f'LR={lr} (Validation)')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png')
    plt.close()

# Generate comparison plot
plot_learning_rate_comparison(results)

# Print final epoch metrics
print("\nFinal Epoch Metrics:")
for lr, res in results.items():
    print(f"\nLearning Rate: {lr}")
    print(f"Final Train Loss: {res['train_losses'][-1]:.4f}")
    print(f"Final Val Loss: {res['val_losses'][-1]:.4f}")
    print(f"Final Train Accuracy: {res['train_accs'][-1]:.2f}%")
    print(f"Final Val Accuracy: {res['val_accs'][-1]:.2f}%")

'''
Final Epoch Metrics:

Learning Rate: 0.001
Final Train Loss: 6.3472
Final Val Loss: 7.3167
Final Train Accuracy: 0.56%
Final Val Accuracy: 0.00%

Learning Rate: 0.0001
Final Train Loss: 5.7042
Final Val Loss: 11.2646
Final Train Accuracy: 1.50%
Final Val Accuracy: 0.75%

Learning Rate: 1e-05
Final Train Loss: 5.7927
Final Val Loss: 9.3959
Final Train Accuracy: 1.32%
Final Val Accuracy: 0.00%

Learning Rate: 1e-06
Final Train Loss: 6.4305
Final Val Loss: 6.5882
Final Train Accuracy: 0.38%
Final Val Accuracy: 0.00%'
'''