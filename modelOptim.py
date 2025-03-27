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
            features = self.feature_extractor(image)
            features = features.view(features.size(0), -1)  # Flatten features
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

# Multimodal Classifier with Configurable Hidden Dimensions
class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim=512, img_dim=512, hidden_dims=None, num_classes=10):
        """
        Args:
            text_dim (int): Dimension of text encoder output
            img_dim (int): Dimension of image encoder output
            hidden_dims (list): List of hidden layer dimensions 
                                If None, defaults to [1024]
            num_classes (int): Number of output classes
        """
        super(MultimodalClassifier, self).__init__()
        self.text_encoder = TextEncoder(text_dim)
        self.image_encoder = ResNetEncoder(img_dim)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [1024]
        
        # Dynamically create hidden layers
        fusion_input_dim = text_dim + img_dim
        self.hidden_layers = nn.ModuleList()
        
        # Create hidden layers
        prev_dim = fusion_input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, text, image):
        # Ensure input tensors are the right shape
        if text.dim() == 3:
            text = text.squeeze(1)  # Remove extra dimension if present
        
        # Extract features
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        
        # Concatenate features
        fused_features = torch.cat((text_features, image_features), dim=1)
        
        # Pass through hidden layers
        x = fused_features
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Final classification
        return self.classifier(x)