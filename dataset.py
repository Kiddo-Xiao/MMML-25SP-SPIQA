import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        """
        Args:
            csv_file (str): CSV file path, including `paper_id`, `question`, `image`, `answer`
            data_dir (str): root path, including `paper_id` folders
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir

        # Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Text Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # `Answer` → `Label` Mapping
        self.answer_to_label = {answer: idx for idx, answer in enumerate(self.data['Answer'].unique())}
        self.label_to_answer = {idx: answer for answer, idx in self.answer_to_label.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        paper_id, question, image_name, answer = row["paper_id"], row["Question"], row["Image"], row["Answer"]

        # Load image
        img_path = os.path.join(self.data_dir, str(paper_id), image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenize question
        question_tokens = self.tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=50)

        # get `Label` corresponding to `Answer`
        label = self.answer_to_label[answer]

        return {
            "image": image,
            "question": question_tokens["input_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
