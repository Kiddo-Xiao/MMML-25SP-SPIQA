import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, models
from PIL import Image
from rouge import Rouge
import math
from tqdm import tqdm


# 定义数据集类
class ImageAnswerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.answer_to_label = {answer: idx for idx, answer in enumerate(self.annotations['Answer'].unique())}
        self.label_to_answer = {idx: answer for answer, idx in self.answer_to_label.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        paper_id = self.annotations.iloc[idx]['paper_id']
        image_name = self.annotations.iloc[idx]['Image']
        answer = self.annotations.iloc[idx]['Answer']
        img_path = os.path.join(self.root_dir, str(paper_id), image_name)
        image = Image.open(img_path).convert('RGB')
        label = self.answer_to_label[answer]
        if self.transform:
            image = self.transform(image)
        return image, label


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据加载
csv_file = 'OverheatData/OverheatData/training_data.csv'
root_dir = 'OverheatData/OverheatData'
dataset = ImageAnswerDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100.0 * correct / total


# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions, references = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predictions.extend([str(p.item()) for p in predicted])
            references.extend([str(l.item()) for l in labels])
    rouge = Rouge().get_scores(' '.join(predictions), ' '.join(references))[0]['rouge-l']['f']
    perplexity = math.exp(running_loss / len(dataloader))
    return running_loss / len(dataloader), 100.0 * correct / total, rouge, perplexity


# 训练多个 ResNet 变体
resnet_variants = [models.resnet18, models.resnet50, models.resnet101]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(dataset.answer_to_label)

for resnet_func in resnet_variants:
    model = resnet_func(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 80
    print(resnet_func)
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    val_loss, val_acc, rouge_score, perplexity = evaluate(model, val_loader, criterion, device)
    print(
        f'Model: {resnet_func.__name__}, Val Acc: {val_acc:.2f}%, ROUGE: {rouge_score:.4f}, Perplexity: {perplexity:.4f}')
