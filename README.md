# Bird Species Classification
A bird image classification project based on Hugging Face Transformers and ViT model, supporting full parameter fine-tuning and LoRA parameter efficient fine-tuning.

## Quick Start

### Installation dependencies
```bash
pip install -r requirements.txt
```

### Full parameter fine-tuning
```bash
python src/train.py --config configs/full_ft.yaml
```

### LoRA fine-tuning
```bash
python src/train_lora.py --config configs/lora.yaml
```

### Model inference
```bash
python src/inference.py \
    --image_path "data/test/aberts_towhee.jpg" \
    --model_path "./outputs/full_ft"
```

