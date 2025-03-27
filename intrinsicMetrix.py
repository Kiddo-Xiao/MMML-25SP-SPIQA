import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import MultimodalDataset
from modelOptim import MultimodalClassifier

# Add ROUGE and Perplexity calculation
from rouge_score import rouge_scorer
import torch.nn.functional as F

def calculate_rouge(predictions, targets, dataset):
    """
    Calculate ROUGE scores between predicted and target answers
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        dataset (MultimodalDataset): Dataset for label-to-answer mapping
    
    Returns:
        dict: ROUGE scores
    """
    # Convert predictions and targets to answers
    pred_answers = [dataset.label_to_answer[pred.item()] for pred in predictions]
    true_answers = [dataset.label_to_answer[target.item()] for target in targets]
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, true in zip(pred_answers, true_answers):
        scores = scorer.score(true, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # Average ROUGE scores
    return {
        key: np.mean(values) for key, values in rouge_scores.items()
    }

def calculate_perplexity(logits, labels):
    """
    Calculate perplexity based on cross-entropy loss
    
    Args:
        logits (torch.Tensor): Model output logits
        labels (torch.Tensor): Ground truth labels
    
    Returns:
        float: Perplexity score
    """
    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits, labels, reduction='mean')
    
    # Convert to perplexity (exp of cross-entropy loss)
    return torch.exp(loss).item()

def train_and_evaluate():
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
        hidden_dims=[2048],  # Single configuration as requested
        num_classes=num_classes
    ).to(device)

    # Optimizer & Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_rouge_scores, val_rouge_scores = [], []
    train_perplexities, val_perplexities = [], []

    num_epochs = 10

    # Train Loop
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        epoch_train_perplexities = []
        epoch_train_rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

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

            # Calculate intrinsic metrics
            perplexity = calculate_perplexity(output_logits, batch["label"])
            epoch_train_perplexities.append(perplexity)

            # ROUGE scores
            rouge_scores = calculate_rouge(preds, batch["label"], dataset)
            for key in rouge_scores:
                epoch_train_rouge_scores[key].append(rouge_scores[key])

        # Aggregate training metrics
        train_loss = total_train_loss / len(train_loader)  
        train_acc = (correct_train / total_train) * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Average training intrinsic metrics
        train_perplexity = np.mean(epoch_train_perplexities)
        train_perplexities.append(train_perplexity)
        
        train_rouge = {key: np.mean(values) for key, values in epoch_train_rouge_scores.items()}
        train_rouge_scores.append(train_rouge)

        # Validation Phase
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        epoch_val_perplexities = []
        epoch_val_rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

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

                # Calculate intrinsic metrics
                perplexity = calculate_perplexity(output_logits, batch["label"])
                epoch_val_perplexities.append(perplexity)

                # ROUGE scores
                rouge_scores = calculate_rouge(preds, batch["label"], dataset)
                for key in rouge_scores:
                    epoch_val_rouge_scores[key].append(rouge_scores[key])

        # Aggregate validation metrics
        val_acc = (correct_val / total_val) * 100
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Average validation intrinsic metrics
        val_perplexity = np.mean(epoch_val_perplexities)
        val_perplexities.append(val_perplexity)
        
        val_rouge = {key: np.mean(values) for key, values in epoch_val_rouge_scores.items()}
        val_rouge_scores.append(val_rouge)

        print(f"Epoch [{epoch+1}/10]:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Train Perplexity: {train_perplexity:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.4f}")
        print(f"  Train ROUGE-1: {train_rouge['rouge1']:.4f}")
        print(f"  Val ROUGE-1: {val_rouge['rouge1']:.4f}")

    # Plotting function for intrinsic metrics
    def plot_intrinsic_metrics(train_perplexities, val_perplexities, train_rouge_scores, val_rouge_scores):
        plt.figure(figsize=(15, 10))
        
        # Perplexity Plot
        plt.subplot(2, 2, 1)
        plt.plot(train_perplexities, label='Train Perplexity')
        plt.plot(val_perplexities, label='Val Perplexity')
        plt.title('Perplexity Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        
        # ROUGE-1 Plot
        plt.subplot(2, 2, 2)
        train_rouge1 = [scores['rouge1'] for scores in train_rouge_scores]
        val_rouge1 = [scores['rouge1'] for scores in val_rouge_scores]
        plt.plot(train_rouge1, label='Train ROUGE-1')
        plt.plot(val_rouge1, label='Val ROUGE-1')
        plt.title('ROUGE-1 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE-1 Score')
        plt.legend()
        
        # ROUGE-2 Plot
        plt.subplot(2, 2, 3)
        train_rouge2 = [scores['rouge2'] for scores in train_rouge_scores]
        val_rouge2 = [scores['rouge2'] for scores in val_rouge_scores]
        plt.plot(train_rouge2, label='Train ROUGE-2')
        plt.plot(val_rouge2, label='Val ROUGE-2')
        plt.title('ROUGE-2 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE-2 Score')
        plt.legend()
        
        # ROUGE-L Plot
        plt.subplot(2, 2, 4)
        train_rougeL = [scores['rougeL'] for scores in train_rouge_scores]
        val_rougeL = [scores['rougeL'] for scores in val_rouge_scores]
        plt.plot(train_rougeL, label='Train ROUGE-L')
        plt.plot(val_rougeL, label='Val ROUGE-L')
        plt.title('ROUGE-L Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE-L Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('intrinsic_metrics_comparison.png')
        plt.close()

    # Generate intrinsic metrics plot
    plot_intrinsic_metrics(train_perplexities, val_perplexities, train_rouge_scores, val_rouge_scores)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'train_rouge_scores': train_rouge_scores,
        'val_rouge_scores': val_rouge_scores
    }

# Run the training with intrinsic metrics
results = train_and_evaluate()

'''
Output1:
Epoch [1/10]:
  Train Loss: 6.4832, Train Acc: 0.00%
  Val Loss: 6.4926, Val Acc: 0.00%
  Train Perplexity: 654.7016
  Val Perplexity: 660.7851
  Train ROUGE-1: 0.1128
  Val ROUGE-1: 0.1110
Epoch [2/10]:
  Train Loss: 6.4734, Train Acc: 0.19%
  Val Loss: 6.5241, Val Acc: 0.00%
  Train Perplexity: 648.3582
  Val Perplexity: 682.1267
  Train ROUGE-1: 0.1025
  Val ROUGE-1: 0.0827
Epoch [3/10]:
  Train Loss: 6.4614, Train Acc: 0.19%
  Val Loss: 6.5731, Val Acc: 0.00%
  Train Perplexity: 640.9945
  Val Perplexity: 716.4657
  Train ROUGE-1: 0.0932
  Val ROUGE-1: 0.0603
Epoch [4/10]:
  Train Loss: 6.4427, Train Acc: 0.38%
  Val Loss: 6.6381, Val Acc: 0.00%
  Train Perplexity: 629.3962
  Val Perplexity: 764.8317
  Train ROUGE-1: 0.0740
  Val ROUGE-1: 0.0393
Epoch [5/10]:
  Train Loss: 6.4184, Train Acc: 0.56%
  Val Loss: 6.7411, Val Acc: 0.00%
  Train Perplexity: 614.5675
  Val Perplexity: 848.3729
  Train ROUGE-1: 0.0479
  Val ROUGE-1: 0.0361
Epoch [6/10]:
  Train Loss: 6.3723, Train Acc: 0.75%
  Val Loss: 6.8505, Val Acc: 0.00%
  Train Perplexity: 587.6670
  Val Perplexity: 947.5994
  Train ROUGE-1: 0.0598
  Val ROUGE-1: 0.0570
Epoch [7/10]:
  Train Loss: 6.3248, Train Acc: 1.13%
  Val Loss: 6.9963, Val Acc: 0.00%
  Train Perplexity: 561.4756
  Val Perplexity: 1098.6274
  Train ROUGE-1: 0.0570
  Val ROUGE-1: 0.0461
Epoch [8/10]:
  Train Loss: 6.2573, Train Acc: 0.94%
  Val Loss: 7.1843, Val Acc: 0.00%
  Train Perplexity: 527.3230
  Val Perplexity: 1330.7245
  Train ROUGE-1: 0.0735
  Val ROUGE-1: 0.0403
Epoch [9/10]:
  Train Loss: 6.1804, Train Acc: 0.75%
  Val Loss: 7.3826, Val Acc: 0.00%
  Train Perplexity: 490.4948
  Val Perplexity: 1632.5913
  Train ROUGE-1: 0.0530
  Val ROUGE-1: 0.0671
Epoch [10/10]:
  Train Loss: 6.1017, Train Acc: 0.56%
  Val Loss: 7.6757, Val Acc: 0.00%
  Train Perplexity: 454.2064
  Val Perplexity: 2203.0544
  Train ROUGE-1: 0.0406
  Val ROUGE-1: 0.0523

Output2:
Epoch [1/10]:
  Train Loss: 6.5018, Train Acc: 0.00%
  Val Loss: 6.5397, Val Acc: 0.00%
  Train Perplexity: 667.0216
  Val Perplexity: 692.8837
  Train ROUGE-1: 0.1178
  Val ROUGE-1: 0.0995
Epoch [2/10]:
  Train Loss: 6.4605, Train Acc: 0.19%
  Val Loss: 6.5868, Val Acc: 0.75%
  Train Perplexity: 639.9226
  Val Perplexity: 726.6990
  Train ROUGE-1: 0.0837
  Val ROUGE-1: 0.0311
Epoch [3/10]:
  Train Loss: 6.4332, Train Acc: 0.75%
  Val Loss: 6.8195, Val Acc: 0.75%
  Train Perplexity: 623.3673
  Val Perplexity: 918.3306
  Train ROUGE-1: 0.0371
  Val ROUGE-1: 0.0302
Epoch [4/10]:
  Train Loss: 6.3766, Train Acc: 0.56%
  Val Loss: 7.1792, Val Acc: 0.75%
  Train Perplexity: 589.0193
  Val Perplexity: 1322.6519
  Train ROUGE-1: 0.0263
  Val ROUGE-1: 0.0074
Epoch [5/10]:
  Train Loss: 6.3335, Train Acc: 0.38%
  Val Loss: 7.6155, Val Acc: 0.00%
  Train Perplexity: 565.5808
  Val Perplexity: 2062.7900
  Train ROUGE-1: 0.0059
  Val ROUGE-1: 0.0000
Epoch [6/10]:
  Train Loss: 6.2845, Train Acc: 0.75%
  Val Loss: 7.9118, Val Acc: 0.00%
  Train Perplexity: 538.3951
  Val Perplexity: 2796.5094
  Train ROUGE-1: 0.0151
  Val ROUGE-1: 0.0000
Epoch [7/10]:
  Train Loss: 6.2419, Train Acc: 0.38%
  Val Loss: 8.1999, Val Acc: 0.00%
  Train Perplexity: 516.6752
  Val Perplexity: 3796.0466
  Train ROUGE-1: 0.0113
  Val ROUGE-1: 0.0006
Epoch [8/10]:
  Train Loss: 6.1745, Train Acc: 0.56%
  Val Loss: 8.5720, Val Acc: 0.00%
  Train Perplexity: 482.5559
  Val Perplexity: 5579.4387
  Train ROUGE-1: 0.0134
  Val ROUGE-1: 0.0000
Epoch [9/10]:
  Train Loss: 6.0846, Train Acc: 0.56%
  Val Loss: 8.7634, Val Acc: 0.00%
  Train Perplexity: 442.3492
  Val Perplexity: 6886.5784
  Train ROUGE-1: 0.0153
  Val ROUGE-1: 0.0000
Epoch [10/10]:
  Train Loss: 5.9986, Train Acc: 0.94%
  Val Loss: 9.4634, Val Acc: 1.49%
  Train Perplexity: 407.7439
  Val Perplexity: 13955.0667
  Train ROUGE-1: 0.0240
  Val ROUGE-1: 0.0253
'''