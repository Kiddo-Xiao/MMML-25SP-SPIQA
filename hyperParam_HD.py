import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MultimodalDataset
from modelOptim import MultimodalClassifier

def train_and_evaluate(hidden_dims):
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
    model = MultimodalClassifier(
        hidden_dims=hidden_dims, 
        num_classes=num_classes
    ).to(device)

    # Optimizer & Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    num_epochs = 10

    # Train Loop
    for epoch in range(num_epochs):
        # Training Phase
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

        # Validation Phase
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

        print(f"Epoch [{epoch+1}/10], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# Configurations to explore
hidden_dim_configs = [
    # Single layer configurations
    [256],
    [512],
    [1024],
    [2048],
    
    # Multi-layer configurations
    [512, 256],
    [1024, 512],
    [2048, 1024],
    
    # More complex configurations
    [512, 256, 128],
    [1024, 512, 256]
]

# Store results
results = {}

# Run experiments
for config in hidden_dim_configs:
    print(f"\nRunning experiment with hidden dimensions: {config}")
    results[tuple(config)] = train_and_evaluate(config)

# Plotting function
def plot_hidden_dim_comparison(results):
    plt.figure(figsize=(15, 10))
    
    # Training Loss
    plt.subplot(2, 2, 1)
    for config, res in results.items():
        plt.plot(res['train_losses'], label=f'Config {config}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation Loss
    plt.subplot(2, 2, 2)
    for config, res in results.items():
        plt.plot(res['val_losses'], label=f'Config {config}')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Training Accuracy
    plt.subplot(2, 2, 3)
    for config, res in results.items():
        plt.plot(res['train_accs'], label=f'Config {config}')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Validation Accuracy
    plt.subplot(2, 2, 4)
    for config, res in results.items():
        plt.plot(res['val_accs'], label=f'Config {config}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hidden_dimensions_comparison.png')
    plt.close()

# Generate comparison plot
plot_hidden_dim_comparison(results)

# Print final epoch metrics
print("\nFinal Epoch Metrics:")
for config, res in results.items():
    print(f"\nHidden Dimensions: {config}")
    print(f"Final Train Loss: {res['train_losses'][-1]:.4f}")
    print(f"Final Val Loss: {res['val_losses'][-1]:.4f}")
    print(f"Final Train Accuracy: {res['train_accs'][-1]:.2f}%")
    print(f"Final Val Accuracy: {res['val_accs'][-1]:.2f}%")

'''
Final Epoch Metrics:

Hidden Dimensions: (256,)
Final Train Loss: 6.0143
Final Val Loss: 7.8311
Final Train Accuracy: 2.07%
Final Val Accuracy: 0.00%

Hidden Dimensions: (512,)
Final Train Loss: 6.2448
Final Val Loss: 7.9924
Final Train Accuracy: 0.75%
Final Val Accuracy: 0.00%

Hidden Dimensions: (1024,)
Final Train Loss: 6.2491
Final Val Loss: 8.3662
Final Train Accuracy: 0.94%
Final Val Accuracy: 0.00%

Hidden Dimensions: (2048,)
Final Train Loss: 5.7904
Final Val Loss: 10.0120
Final Train Accuracy: 0.94%
Final Val Accuracy: 1.49%

Hidden Dimensions: (512, 256)
Final Train Loss: 6.2867
Final Val Loss: 7.5157
Final Train Accuracy: 0.19%
Final Val Accuracy: 0.00%

Hidden Dimensions: (1024, 512)
Final Train Loss: 6.2592
Final Val Loss: 8.2385
Final Train Accuracy: 0.56%
Final Val Accuracy: 0.00%

Hidden Dimensions: (2048, 1024)
Final Train Loss: 6.2651
Final Val Loss: 8.6998
Final Train Accuracy: 0.75%
Final Val Accuracy: 0.00%

Hidden Dimensions: (512, 256, 128)
Final Train Loss: 6.3946
Final Val Loss: 7.1291
Final Train Accuracy: 0.19%
Final Val Accuracy: 0.00%

Hidden Dimensions: (1024, 512, 256)
Final Train Loss: 6.3462
Final Val Loss: 7.4648
Final Train Accuracy: 0.19%
Final Val Accuracy: 0.00%
'''