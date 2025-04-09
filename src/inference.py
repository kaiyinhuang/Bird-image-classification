from transformers import ViTForImageClassification, ViTImageProcessor, pipeline
from peft import PeftModel
from data_loader import load_image, get_transforms
import torch
import os

def predict(image_path: str, model_path: str):
    # 加载模型
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA 模型：加载基础模型 + 适配器
        base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
    else:
        # 全参数模型：直接加载
        model = ViTForImageClassification.from_pretrained(model_path)
    
    # 预处理图像
    transform = get_transforms("val")
    image = load_image({"image_file_path": image_path}, transform)["image"]
    
    # 推理
    processor = ViTImageProcessor.from_pretrained(model_path)
    inputs = processor(images=image.unsqueeze(0), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = model.config.id2label[outputs.logits.argmax().item()]
    return predicted_class

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model_path", type=str, default="./outputs/full_ft", help="模型路径")
    args = parser.parse_args()
    
    print("预测结果:", predict(args.image_path, args.model_path))
