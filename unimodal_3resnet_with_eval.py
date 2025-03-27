import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from rouge import Rouge
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        return image, label, img_path  # Added img_path to return


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
    for images, labels, _ in dataloader:  # Updated to unpack img_path
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


# 评估函数和失败案例收集
def evaluate(model, dataloader, criterion, device, model_name):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions, references = [], []
    failed_examples = []  # To store failed examples

    with torch.no_grad():
        for images, labels, img_paths in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            predictions.extend([str(p.item()) for p in predicted])
            references.extend([str(l.item()) for l in labels])

            # Collect failed examples
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    failed_examples.append({
                        'image_path': img_paths[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'true_answer': dataset.label_to_answer[labels[i].item()],
                        'predicted_answer': dataset.label_to_answer[predicted[i].item()],
                        'model_name': model_name
                    })

    # Save failed examples to a CSV file
    if failed_examples:
        failed_df = pd.DataFrame(failed_examples)
        os.makedirs('failed_examples', exist_ok=True)
        failed_df.to_csv(f'failed_examples/failed_{model_name}.csv', index=False)

        # Save some example images
        os.makedirs(f'failed_examples/images_{model_name}', exist_ok=True)
        for i, example in enumerate(failed_examples[:10]):  # Save first 10 examples
            img = Image.open(example['image_path'])
            plt.figure()
            plt.imshow(img)
            plt.title(f"True: {example['true_answer']}\nPredicted: {example['predicted_answer']}")
            plt.savefig(f"failed_examples/images_{model_name}/failed_example_{i}.png")
            plt.close()

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
    model_name = resnet_func.__name__
    print(f"\nTraining {model_name}")

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    val_loss, val_acc, rouge_score, perplexity = evaluate(model, val_loader, criterion, device, model_name)
    print(
        f'Model: {model_name}, Val Acc: {val_acc:.2f}%, ROUGE: {rouge_score:.4f}, Perplexity: {perplexity:.4f}')