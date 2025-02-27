import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SPIQADataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer_name="bert-base-uncased", transform=None, max_length=128):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            root_dir: Directory with all the images organized in paper_id folders
            tokenizer_name: Name of the pretrained tokenizer to use
            transform: Optional transform to be applied on images
            max_length: Max sequence length for tokenizer
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Encode answers as labels
        self.label_encoder = LabelEncoder()
        self.data['answer_encoded'] = self.label_encoder.fit_transform(self.data['answer'])
        self.num_classes = len(self.label_encoder.classes_)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        paper_id = self.data.iloc[idx]['paper_id']
        image_name = self.data.iloc[idx]['image_name']
        question = self.data.iloc[idx]['question']
        answer_encoded = self.data.iloc[idx]['answer_encoded']
        
        # Create image path
        img_path = os.path.join(self.root_dir, str(paper_id), image_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize question
        question_encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Flatten tensors
        input_ids = question_encoding['input_ids'].squeeze()
        attention_mask = question_encoding['attention_mask'].squeeze()
        
        # Create a sample with the paper_id as string for reference
        sample = {
            'paper_id': str(paper_id),
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(answer_encoded, dtype=torch.long)
        }
        
        return sample
    
    def get_answer_from_encoding(self, encoding):
        """Convert encoded label back to text answer"""
        return self.label_encoder.inverse_transform([encoding])[0]


def create_data_loaders(csv_file, root_dir, tokenizer_name="bert-base-uncased", 
                        transform=None, batch_size=16, test_size=0.2, random_state=42):
    """Create train and test data loaders with a given split ratio"""
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    full_dataset = SPIQADataset(csv_file, root_dir, tokenizer_name, transform)
    
    # Split dataset
    train_indices, test_indices = train_test_split(
        np.arange(len(full_dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=full_dataset.data['answer_encoded']  # Ensure balanced classes
    )
    
    # Create sub-datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, full_dataset 