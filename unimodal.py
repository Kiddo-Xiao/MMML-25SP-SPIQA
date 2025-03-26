import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 自定义数据集类
class ImageAnswerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): CSV 文件的路径，包含 `paper_id`, `image`, `answer`。
            root_dir (str): 根目录，包含以 `paper_id` 命名的文件夹。
            transform (callable, optional): 可选的图像预处理操作。
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # 将 answer 转换为类别标签
        self.answer_to_label = {answer: idx for idx, answer in enumerate(self.annotations['Answer'].unique())}
        self.label_to_answer = {idx: answer for answer, idx in self.answer_to_label.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取 CSV 中的记录
        paper_id = self.annotations.iloc[idx]['paper_id']
        image_name = self.annotations.iloc[idx]['Image']
        answer = self.annotations.iloc[idx]['Answer']

        # 构建图片路径
        img_path = os.path.join(self.root_dir, str(paper_id), image_name)
        image = Image.open(img_path).convert('RGB')  # 确保图片是 RGB 格式

        # 将 answer 转换为标签
        label = self.answer_to_label[answer]

        # 图像预处理
        if self.transform:
            image = self.transform(image)
        return image, label

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为 Tensor
])

# 创建数据集
csv_file = 'OverheatData/OverheatData/training_data.csv'  # 替换为你的 CSV 文件路径
root_dir = 'OverheatData/OverheatData'  # 替换为你的根目录路径
dataset = ImageAnswerDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# 创建 DataLoader
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 ResNet 模型并修改最后一层
num_classes = len(dataset.answer_to_label)  # 类别数为 answer 的唯一值数量
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # 修改最后一层

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(resnet.parameters(), lr=0.001)

# 训练设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(dataloader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

# 训练和验证
num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(resnet, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(resnet, val_loader, criterion, device)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# 保存模型
torch.save(resnet.state_dict(), 'resnet_model.pth')