import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from rouge import Rouge
import math


# 自定义数据集类
class TextAnswerDataset(Dataset):
    def __init__(self, csv_file, tokenizer, vocab, max_length=50):
        self.annotations = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.answer_to_label = {answer: idx for idx, answer in enumerate(self.annotations['Answer'].unique())}
        self.label_to_answer = {idx: answer for answer, idx in self.answer_to_label.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        question = self.annotations.iloc[idx]['Question']
        answer = self.annotations.iloc[idx]['Answer']
        tokens = self.tokenizer(question)[:self.max_length]
        indices = [self.vocab[token] for token in tokens]
        indices += [0] * (self.max_length - len(indices))
        label = self.answer_to_label[answer]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 分词器
tokenizer = get_tokenizer('basic_english')

# 读取 CSV 文件并构建词汇表
csv_file = 'OverheatData/OverheatData/training_data.csv'
df = pd.read_csv(csv_file)
vocab = build_vocab_from_iterator((tokenizer(text) for text in df['Question']), specials=['<pad>', '<unk>'])
vocab.set_default_index(vocab['<unk>'])

# 数据加载
dataset = TextAnswerDataset(csv_file=csv_file, tokenizer=tokenizer, vocab=vocab)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# 定义不同深度的文本分类模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        x = self.relu(self.fc1(embedded))
        return self.fc2(x)


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    predictions, references = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            predictions.extend(map(str, outputs.argmax(1).tolist()))
            references.extend(map(str, labels.tolist()))
    rouge = Rouge().get_scores(' '.join(predictions), ' '.join(references))[0]['rouge-l']['f']
    perplexity = math.exp(running_loss / len(dataloader))
    return running_loss / len(dataloader), 100.0 * correct / total, rouge, perplexity


# 训练不同深度的模型
model_variants = [(100, 50), (200, 100), (300, 150)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(dataset.answer_to_label)

for embed_dim, hidden_dim in model_variants:
    model = TextClassifier(len(vocab), embed_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    num_epochs = 80
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    val_loss, val_acc, rouge_score, perplexity = evaluate(model, val_loader, criterion, device)
    print(
        f'Model (Embed {embed_dim}, Hidden {hidden_dim}): Val Acc: {val_acc:.2f}%, ROUGE: {rouge_score:.4f}, Perplexity: {perplexity:.4f}')