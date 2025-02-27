import argparse
import torch
import json
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from model import MultimodalClassifier
from torch.cuda.amp import autocast

def predict(args):
    # Set device with specified GPU if provided
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # Load label encoder
    with open('label_encoder.json', 'r') as f:
        class_labels = json.load(f)
    num_classes = len(class_labels)
    
    # Initialize model
    model = MultimodalClassifier(
        num_classes=num_classes,
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        hidden_size=args.hidden_size
    )
    
    # Load model weights
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(args.image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Prepare question
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    question_encoding = tokenizer(
        args.question,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    question = {k: v.to(device) for k, v in question_encoding.items()}
    
    # Prepare paper ID by extracting the numeric part
    paper_id_num = int(args.paper_id.split('.')[0])
    paper_id = torch.tensor([[paper_id_num]], dtype=torch.long).to(device)
    
    # Get prediction with mixed precision if available
    with torch.no_grad():
        if args.mixed_precision and torch.cuda.is_available():
            with autocast():
                outputs = model(image, question, paper_id)
        else:
            outputs = model(image, question, paper_id)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    predicted_answer = class_labels[prediction.item()]
    confidence = confidence.item()
    
    print(f"\nQuestion: {args.question}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Confidence: {confidence:.4f}")
    
    # Get top 3 predictions
    if len(class_labels) > 1:
        print("\nTop 3 predictions:")
        top_probs, top_preds = torch.topk(probabilities, min(3, len(class_labels)))
        for i in range(top_probs.size(1)):
            print(f"{class_labels[top_preds[0][i].item()]}: {top_probs[0][i].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with the SPIQA model")
    parser.add_argument("--paper_id", type=str, required=True, 
                        help="Paper ID")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to the image")
    parser.add_argument("--question", type=str, required=True, 
                        help="Question about the image")
    parser.add_argument("--text_model", type=str, default="bert-base-uncased", 
                        help="Pre-trained text model used in training")
    parser.add_argument("--vision_model", type=str, default="google/vit-base-patch16-224", 
                        help="Pre-trained vision model used in training")
    parser.add_argument("--hidden_size", type=int, default=512, 
                        help="Hidden size in fusion layers")
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Size images were resized to during training")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Specific GPU to use (e.g. 0, 1)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision for faster inference")
    
    args = parser.parse_args()
    predict(args) 