import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms

class SPIQADataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, transform=None, max_length=128):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            tokenizer: Tokenizer for processing questions.
            transform (callable, optional): Optional transform to be applied on an image.
            max_length (int): Maximum length of tokenized question.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        # Encode the answers to numerical labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data_frame['Answer'])
        self.num_classes = len(self.label_encoder.classes_)
        
        # Create a mapping of paper IDs to integers
        unique_paper_ids = self.data_frame['paper_id'].unique()
        self.paper_id_to_idx = {pid: idx for idx, pid in enumerate(unique_paper_ids)}
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        paper_id = self.data_frame.iloc[idx]['paper_id']
        image_name = self.data_frame.iloc[idx]['Image']
        question = self.data_frame.iloc[idx]['Question']
        label = self.labels[idx]
        
        # Extract only the numeric part before the first dot
        paper_id_num = int(paper_id.split('.')[0])
        
        # Construct image path (still using the original paper_id for the path)
        img_path = os.path.join(self.root_dir, str(paper_id), image_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize question
        question_encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        question_encoding = {k: v.squeeze(0) for k, v in question_encoding.items()}
        
        # Use the numeric paper_id for the feature
        paper_id_feature = torch.tensor([paper_id_num], dtype=torch.long)
        
        return {
            'image': image,
            'question': question_encoding,
            'paper_id': paper_id_feature,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(csv_path, image_root_dir, tokenizer_name="bert-base-uncased", batch_size=16, test_size=0.2, 
                     image_size=224, random_state=42, num_workers=4, pin_memory=True):
    """
    Create train and test data loaders
    
    Args:
        csv_path: Path to the CSV file with annotations
        image_root_dir: Directory with all the images
        tokenizer_name: Name of the pretrained tokenizer to use
        batch_size: Batch size for training
        test_size: Proportion of data to use for testing
        image_size: Size to resize images to
        random_state: Random seed for reproducibility
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
    """
    from sklearn.model_selection import train_test_split
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Read and split the dataset
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save splits to CSV for reference
    train_df.to_csv('train_split.csv', index=False)
    test_df.to_csv('test_split.csv', index=False)
    
    # Create temporary CSV files for the splits
    train_csv = 'temp_train.csv'
    test_csv = 'temp_test.csv'
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Create datasets
    train_dataset = SPIQADataset(train_csv, image_root_dir, tokenizer, transform)
    test_dataset = SPIQADataset(test_csv, image_root_dir, tokenizer, transform)
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(test_csv)
    
    # Create data loaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, test_loader, train_dataset.num_classes, train_dataset.label_encoder 