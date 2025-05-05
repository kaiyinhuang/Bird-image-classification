import torch
from torchvision import models, transforms
from peft import PeftModel
from PIL import Image
import json
import argparse
import os
from data_utils import get_transforms # 从 data_utils.py 导入

def load_model(adapter_dir, base_model_name="resnet50", num_classes=200, device="cuda"):
    """Loads the base model and applies LoRA adapters."""
    print(f"Loading base model ({base_model_name})...")
    # TODO: Add flexibility for different base models if needed
    if base_model_name == "resnet50":
        model_base = models.resnet50(weights=None) # Load architecture
        num_ftrs = model_base.fc.in_features
        model_base.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    print(f"Loading LoRA adapters from {adapter_dir}...")
    # Load the LoRA model
    model_lora = PeftModel.from_pretrained(model_base, adapter_dir)
    model_lora = model_lora.to(device)
    model_lora.eval()
    print("Model loaded and set to evaluation mode.")
    return model_lora

def predict(image_path, model, class_names, device):
    """Predicts the class for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None

    # Get validation transforms (important: use the same as during evaluation)
    data_transforms = get_transforms()
    transform = data_transforms['val'] # 使用验证集的变换
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device) # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(batch_t)
        # Assuming standard output, adjust if model output is different
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probabilities = torch.softmax(logits, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    predicted_class_idx = preds.item()
    predicted_class_name = class_names[predicted_class_idx]
    confidence_score = confidence.item()

    return predicted_class_name, confidence_score

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- Load Class Names ---
    print(f"Loading class names from {args.class_names_path}...")
    try:
        with open(args.class_names_path, 'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names.")
    except Exception as e:
        print(f"Error loading class names: {e}")
        return

    # --- Load Model ---
    model = load_model(args.lora_adapter_dir, num_classes=num_classes, device=device)

    # --- Predict ---
    print(f"Predicting image: {args.image_path}...")
    predicted_class, confidence = predict(args.image_path, model, class_names, device)

    # --- Output Result ---
    if predicted_class:
        print("-" * 30)
        print(f"Prediction Result:")
        print(f"  Image: {os.path.basename(args.image_path)}")
        print(f"  Predicted Bird: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict bird species from an image using a LoRA model.')
    parser.add_argument('--lora_adapter_dir', type=str, required=True, help='Directory containing the saved LoRA adapter files (adapter_model.bin, etc.).')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input bird image.')
    parser.add_argument('--class_names_path', type=str, required=True, help='Path to the class_names.json file.')
    # parser.add_argument('--base_model_name', type=str, default='resnet50', help='Name of the base model used during training.') # 可选，如果支持多种基础模型
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')

    args = parser.parse_args()
    main(args)
