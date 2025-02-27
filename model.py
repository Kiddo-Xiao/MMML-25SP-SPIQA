import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, text_model_name="bert-base-uncased", 
                 vision_model_name="google/vit-base-patch16-224", hidden_size=512):
        super(MultimodalClassifier, self).__init__()
        
        # Text encoder (BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_embedding_dim = self.text_encoder.config.hidden_size
        
        # Vision encoder (ViT)
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        vision_embedding_dim = self.vision_encoder.config.hidden_size
        
        # Paper ID embedding
        self.paper_id_embedding = nn.Embedding(10000, 64)  # Assuming max 10000 papers
        
        # Fusion layer - use efficient implementation for GPU
        self.fusion = nn.Sequential(
            nn.Linear(text_embedding_dim + vision_embedding_dim + 64, hidden_size),
            nn.SiLU(),  # More GPU-efficient than ReLU
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, image, question, paper_id):
        # Extract image features
        vision_output = self.vision_encoder(image)
        vision_features = vision_output.pooler_output  # [batch_size, vision_dim]
        
        # Extract text features
        text_output = self.text_encoder(**question)
        text_features = text_output.pooler_output  # [batch_size, text_dim]
        
        # Extract paper ID features
        paper_id_features = self.paper_id_embedding(paper_id).squeeze(1)  # [batch_size, 64]
        
        # Combine features
        combined_features = torch.cat([vision_features, text_features, paper_id_features], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits 