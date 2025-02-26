import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


# 自定义数据集类
class TextAnswerDataset(Dataset):
    def __init__(self, csv_file, tokenizer, vocab, max_length=50):
        """
        Args:
            csv_file (str): CSV 文件的路径，包含 `question` 和 `answer`。
            tokenizer: 用于将文本转换为 token 的分词器。
            vocab: 词汇表，用于将 token 转换为索引。
            max_length (int): 文本的最大长度，超过的部分将被截断。
        """
        self.annotations = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

        # 将 answer 转换为类别标签
        self.answer_to_label = {answer: idx for idx, answer in enumerate(self.annotations['Answer'].unique())}
        self.label_to_answer = {idx: answer for answer, idx in self.answer_to_label.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取 CSV 中的记录
        question = self.annotations.iloc[idx]['Question']
        answer = self.annotations.iloc[idx]['Answer']

        # 将 question 转换为 token 并截断到最大长度
        tokens = self.tokenizer(question)[:self.max_length]
        # 将 token 转换为索引
        indices = [self.vocab[token] for token in tokens]
        # 填充到最大长度
        indices += [0] * (self.max_length - len(indices))  # 0 是填充符号

        # 将 answer 转换为标签
        label = self.answer_to_label[answer]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 分词器
tokenizer = get_tokenizer('basic_english')


# 构建词汇表
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


# 读取 CSV 文件并构建词汇表
csv_file = 'OverheatData/OverheatData/training_data.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file)
vocab = build_vocab_from_iterator(yield_tokens(df['Question']), specials=['<pad>', '<unk>'])
vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为未知 token

# 创建数据集
dataset = TextAnswerDataset(csv_file=csv_file, tokenizer=tokenizer, vocab=vocab)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建 DataLoader
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 定义文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, max_length)
        embedded = self.embedding(x)  # (batch_size, max_length, embed_dim)
        # 取每个句子的平均嵌入
        pooled = embedded.mean(dim=1)  # (batch_size, embed_dim)
        # 全连接层
        output = self.fc(pooled)  # (batch_size, num_classes)
        return output


# 初始化模型
vocab_size = len(vocab)
embed_dim = 100  # 嵌入维度
num_classes = len(dataset.answer_to_label)  # 类别数为 answer 的唯一值数量
model = TextClassifier(vocab_size, embed_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 训练设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
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
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
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
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'text_model.pth')
