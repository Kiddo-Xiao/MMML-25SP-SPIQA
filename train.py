import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import get_data_loaders
from model import MultimodalClassifier

def train_model(args):
    # Set up distributed training if multiple GPUs are available
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for distributed training")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
    
    print(f"Using device: {device} (GPU name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    
    # Pre-check GPU memory
    if torch.cuda.is_available():
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")
    
    # Initialize mixed precision training
    scaler = GradScaler() if args.mixed_precision else None
    
    # Get data loaders with optimized settings for GPU
    train_loader, test_loader, num_classes, label_encoder = get_data_loaders(
        args.csv_path, 
        args.image_root_dir,
        args.text_model,
        args.batch_size,
        args.test_size,
        args.image_size,
        num_workers=args.num_workers,
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    # Initialize model
    model = MultimodalClassifier(
        num_classes=num_classes,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        hidden_size=args.hidden_size
    )
    model.to(device)
    
    # Wrap model with DDP for multi-GPU training
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', disable=local_rank != 0)
        
        for batch in train_pbar:
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            question = {k: v.to(device, non_blocking=True) for k, v in batch['question'].items()}
            paper_id = batch['paper_id'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if args.mixed_precision:
                with autocast():
                    outputs = model(image, question, paper_id)
                    loss = criterion(outputs, labels)
                
                # Backward and optimize with gradient scaling
                scaler.scale(loss).backward()
                # Gradient clipping to prevent exploding gradients
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image, question, paper_id)
                loss = criterion(outputs, labels)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
            
            # Periodically free memory
            if torch.cuda.is_available() and args.gpu_cleanup_freq > 0 and (train_pbar.n % args.gpu_cleanup_freq) == 0:
                torch.cuda.empty_cache()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Test]', disable=local_rank != 0)
            for batch in test_pbar:
                image = batch['image'].to(device, non_blocking=True)
                question = {k: v.to(device, non_blocking=True) for k, v in batch['question'].items()}
                paper_id = batch['paper_id'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                # Use mixed precision for inference too
                if args.mixed_precision:
                    with autocast():
                        outputs = model(image, question, paper_id)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(image, question, paper_id)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                test_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average test loss and accuracy
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        test_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Only print on main process for distributed training
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
            
            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                
                # Save model
                if args.multi_gpu:
                    model.module.state_dict()
                else:
                    model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, 'best_model.pth')
                
                # Save label encoder
                with open('label_encoder.json', 'w') as f:
                    json.dump(label_encoder.classes_.tolist(), f)
    
    # Final evaluation on main process
    if local_rank == 0:
        checkpoint = torch.load('best_model.pth', weights_only=True)
        if args.multi_gpu:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                image = batch['image'].to(device, non_blocking=True)
                question = {k: v.to(device, non_blocking=True) for k, v in batch['question'].items()}
                paper_id = batch['paper_id'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                outputs = model(image, question, paper_id)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Get unique classes present in the test set
        unique_classes = np.sort(np.unique(np.concatenate([np.unique(all_labels), np.unique(all_preds)])))
        class_names = [label_encoder.classes_[i] for i in unique_classes]
        
        # Print classification report with only classes that appear in the test set
        print("\nClassification Report:")
        try:
            print(classification_report(
                all_labels, 
                all_preds,
                labels=unique_classes,
                target_names=class_names
            ))
        except Exception as e:
            print(f"Error generating classification report: {e}")
            print(f"Number of unique classes in predictions: {len(np.unique(all_preds))}")
            print(f"Number of unique classes in labels: {len(np.unique(all_labels))}")
            print(f"Total unique classes in test set: {len(unique_classes)}")
            print(f"Total classes in label_encoder: {len(label_encoder.classes_)}")
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.epochs + 1), test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        
        print(f"Best test accuracy: {best_accuracy:.4f}")
        print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multimodal classifier for SPIQA dataset")
    parser.add_argument("--csv_path", type=str, default="OverheatData/training_data.csv", 
                        help="Path to the CSV file with annotations")
    parser.add_argument("--image_root_dir", type=str, default="OverheatData", 
                        help="Directory with all the images")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", 
                        help="Pre-trained text model to use")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224", 
                        help="Pre-trained vision model to use")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data to use for testing")
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Size to resize images to")
    parser.add_argument("--hidden_size", type=int, default=512, 
                        help="Hidden size in fusion layers")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs to train for")
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Use mixed precision training for faster computation")
    parser.add_argument("--multi_gpu", action="store_true", 
                        help="Use multiple GPUs for distributed training")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--gpu_cleanup_freq", type=int, default=200, 
                        help="Frequency of GPU memory cleanup (0 to disable)")
    
    args = parser.parse_args()
    train_model(args) 