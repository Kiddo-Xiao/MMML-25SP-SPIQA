import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MultimodalDataset
from model import MultimodalClassifier
'''
Weight Decay Hyperparameter Sweep
'''
def run_experiment(weight_decay):
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

    # Optimizer with varying weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    weight_norms = []  # Track model weight norms

    num_epochs = 10

    # Train Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        # Calculate initial weight norm
        initial_weight_norm = sum(p.data.norm(2).item() for p in model.parameters())
        weight_norms.append(initial_weight_norm)

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
        'val_accs': val_accs,
        'weight_norms': weight_norms
    }

# Weight decay values to test
weight_decays = [0, 1e-5, 1e-4, 1e-3]

# Store results
results = {}

# Run experiments
for wd in weight_decays:
    print(f"\nRunning experiment with weight decay: {wd}")
    results[wd] = run_experiment(wd)

# Plotting function
def plot_weight_decay_comparison(results):
    plt.figure(figsize=(15, 12))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    for wd, res in results.items():
        plt.plot(res['train_losses'], label=f'Weight Decay={wd} (Train)')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation Loss
    plt.subplot(2, 2, 2)
    for wd, res in results.items():
        plt.plot(res['val_losses'], label=f'Weight Decay={wd} (Validation)')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Training Accuracy
    plt.subplot(2, 2, 3)
    for wd, res in results.items():
        plt.plot(res['train_accs'], label=f'Weight Decay={wd} (Train)')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Weight Norms
    plt.subplot(2, 2, 4)
    for wd, res in results.items():
        plt.plot(res['weight_norms'], label=f'Weight Decay={wd}')
    plt.title('Model Weight Norms')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weight_decay_comparison.png')
    plt.close()

# Generate comparison plot
plot_weight_decay_comparison(results)

# Print final epoch metrics
print("\nFinal Epoch Metrics:")
for wd, res in results.items():
    print(f"\nWeight Decay: {wd}")
    print(f"Final Train Loss: {res['train_losses'][-1]:.4f}")
    print(f"Final Val Loss: {res['val_losses'][-1]:.4f}")
    print(f"Final Train Accuracy: {res['train_accs'][-1]:.2f}%")
    print(f"Final Val Accuracy: {res['val_accs'][-1]:.2f}%")
    print(f"Final Weight Norm: {res['weight_norms'][-1]:.4f}")


'''
Final Epoch Metrics:

Weight Decay: 0
Final Train Loss: 6.0418
Final Val Loss: 9.1385
Final Train Accuracy: 1.50%
Final Val Accuracy: 0.00%
Final Weight Norm: 4862.9296

Weight Decay: 1e-05
Final Train Loss: 6.1908
Final Val Loss: 8.5876
Final Train Accuracy: 1.32%
Final Val Accuracy: 0.00%
Final Weight Norm: 4862.8353

Weight Decay: 0.0001
Final Train Loss: 5.7830
Final Val Loss: 9.4812
Final Train Accuracy: 1.13%
Final Val Accuracy: 0.00%
Final Weight Norm: 4863.2246

Weight Decay: 0.001
Final Train Loss: 6.2046
Final Val Loss: 8.5463
Final Train Accuracy: 1.69%
Final Val Accuracy: 0.75%
Final Weight Norm: 4862.7658'
'''