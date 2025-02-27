# SPIQA - Scientific Paper Image Question Answering

A multimodal classifier that predicts answers to questions about scientific paper images.

## Dataset Structure
- `OverheatData/training_data.csv`: Contains question-answer pairs with paper_id and image_name
- `OverheatData/{paper_id}/`: Folders containing images for each paper

## Model Architecture

The SPIQA model is a multimodal classifier that integrates information from three sources:

### Input Components
1. **Image Encoder**: A Vision Transformer (ViT) processes the scientific paper image to extract visual features.
   - Pre-trained model: `google/vit-base-patch16-224`
   - Input size: 224×224 RGB images
   - Output: 768-dimensional image embedding

2. **Question Encoder**: A BERT-based language model processes the question text.
   - Pre-trained model: `bert-base-uncased`
   - Input: Tokenized question with max length of 128
   - Output: 768-dimensional text embedding

3. **Paper ID Embedding**: An embedding layer maps the numerical paper ID to a dense representation.
   - Input: Paper ID (extracted from the paper_id string)
   - Embedding dimension: 64

### Fusion and Classification
The three embeddings (image, question, and paper ID) are concatenated into a unified representation vector. This combined representation is then processed through:

1. **Fusion Layers**: Two fully-connected layers with SiLU activations and dropout:
   - Linear layer: (768 + 768 + 64) → 512
   - SiLU activation + Dropout (0.2)
   - Linear layer: 512 → 512
   - SiLU activation + Dropout (0.2)

2. **Classification Head**: A final linear layer maps the fused representation to answer classes:
   - Linear layer: 512 → num_classes

The model is trained end-to-end using cross-entropy loss and AdamW optimizer. Mixed precision training with automatic gradient scaling is supported for better GPU efficiency.

## GPU Optimization Features

- **Mixed Precision Training**: Uses FP16 precision where appropriate to speed up training
- **Distributed Training**: Supports multi-GPU training with DistributedDataParallel
- **Efficient Data Loading**: Optimized data loading with pin_memory and multiple workers
- **Memory Management**: Periodic GPU cache clearing to prevent out-of-memory errors
- **Gradient Clipping**: Prevents exploding gradients during training

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run training: `python train.py --mixed_precision`
3. Run inference: `python predict.py --paper_id {id} --image {image_path} --question "Your question here"`

### Training with Multiple GPUs
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py --multi_gpu --mixed_precision
```

### Command-Line Arguments
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--mixed_precision`: Enable mixed precision training
- `--num_workers`: Number of data loading workers (default: 4)
- `--gpu`: Specify GPU ID for inference