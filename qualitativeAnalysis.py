import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import MultimodalDataset
from modelOptim import MultimodalClassifier

# Add ROUGE and Perplexity calculation
from rouge_score import rouge_scorer
import torch.nn.functional as F

# Previously defined functions remain the same (calculate_rouge, calculate_perplexity)

def train_and_evaluate(print_examples=True):
    # Previous setup code remains the same
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
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

    # Qualitative Analysis Preparation
    qualitative_examples = []

    # Open file for writing qualitative analysis
    with open('qualitative_analysis.txt', 'w') as f:
        f.write("Multimodal Classification Qualitative Analysis\n")
        f.write("=============================================\n\n")

        # Train Loop
        for epoch in range(num_epochs):
            # Training phase remains the same
            # ... (previous training code)

            # Qualitative Analysis: Collect some example predictions
            if print_examples and (epoch == 0 or epoch == num_epochs - 1):
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        if batch_idx >= 5:  # Limit to first 5 batches
                            break

                        batch["question"] = batch["question"].to(device)
                        batch["image"] = batch["image"].to(device)
                        batch["label"] = batch["label"].to(device)

                        output_logits = model(batch["question"], batch["image"])
                        preds = torch.argmax(output_logits, dim=-1)

                        # Prepare qualitative examples
                        for i in range(len(preds)):
                            # Decode back to original data
                            true_label = batch["label"][i].cpu().item()
                            pred_label = preds[i].cpu().item()
                            
                            # Convert back to answer strings
                            true_answer = dataset.label_to_answer[true_label]
                            pred_answer = dataset.label_to_answer[pred_label]

                            # Get original question and paper_id
                            original_data = dataset.data.iloc[val_dataset.indices[batch_idx * BATCH_SIZE + i]]
                            question = original_data["Question"]
                            paper_id = original_data["paper_id"]
                            image_name = original_data["Image"]

                            # Write to file
                            f.write(f"\n--- Epoch {epoch + 1} Example ---\n")
                            f.write(f"Paper ID: {paper_id}\n")
                            f.write(f"Question: {question}\n")
                            f.write(f"Image: {image_name}\n")
                            f.write(f"True Answer: {true_answer}\n")
                            f.write(f"Predicted Answer: {pred_answer}\n")
                            f.write(f"Prediction Confidence: {torch.softmax(output_logits[i], dim=-1).max().cpu().item():.4f}\n")

                            qualitative_examples.append({
                                'epoch': epoch + 1,
                                'paper_id': paper_id,
                                'question': question,
                                'image_name': image_name,
                                'true_answer': true_answer,
                                'predicted_answer': pred_answer,
                                'confidence': torch.softmax(output_logits[i], dim=-1).max().cpu().item()
                            })

        # Write a summary section
        f.write("\n\n--- SUMMARY OF QUALITATIVE ANALYSIS ---\n")
        f.write(f"Total Examples Analyzed: {len(qualitative_examples)}\n")
        
        # Compute some basic statistics
        correct_predictions = sum(1 for ex in qualitative_examples if ex['true_answer'] == ex['predicted_answer'])
        f.write(f"Correct Predictions: {correct_predictions} / {len(qualitative_examples)}\n")
        f.write(f"Accuracy: {correct_predictions / len(qualitative_examples) * 100:.2f}%\n")
        
        # Confidence analysis
        confidences = [ex['confidence'] for ex in qualitative_examples]
        f.write(f"Average Prediction Confidence: {np.mean(confidences):.4f}\n")
        f.write(f"Max Prediction Confidence: {np.max(confidences):.4f}\n")
        f.write(f"Min Prediction Confidence: {np.min(confidences):.4f}\n")

    print("Qualitative analysis saved to qualitative_analysis.txt")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'train_rouge_scores': train_rouge_scores,
        'val_rouge_scores': val_rouge_scores,
        'qualitative_examples': qualitative_examples
    }

# Run the training with intrinsic metrics
results = train_and_evaluate()