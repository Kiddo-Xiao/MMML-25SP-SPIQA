import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

# Image Encoder
class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, image):
        with torch.no_grad():
            features = self.feature_extractor(image).squeeze()
        return self.fc(features)

# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids, return_dict=True)
        return self.fc(outputs.pooler_output)

# Multimodal Classifier
class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim=512, img_dim=512, hidden_dim=1024, num_classes=10):
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = TextEncoder(text_dim)
        self.image_encoder = ResNetEncoder(img_dim)
        self.fc1 = nn.Linear(text_dim + img_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text, image):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        fused_features = torch.cat((text_features, image_features), dim=1)
        out = self.relu(self.fc1(fused_features))
        return self.fc2(out)  # 直接输出 logits
