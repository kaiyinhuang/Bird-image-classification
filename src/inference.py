from transformers import ViTForImageClassification, ViTImageProcessor, pipeline
from .data_preprocessing import load_image, get_transforms
import torch

def load_model(model_path):
    # 检查是否存在 LoRA 适配器配置
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # 加载基础模型
        base_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=200,
            ignore_mismatched_sizes=True
        )
        # 加载 LoRA 适配器并合并权重
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合并到基础模型
    else:
        # 加载全参数微调模型
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)

    # 预处理图像
    transform = get_transforms("val")
    image = load_image(image_path, transform)
    
    # 推理
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = model.config.id2label[outputs.logits.argmax().item()]
    return pred_label

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./outputs/full_ft")
    args = parser.parse_args()
    
    print("预测结果:", predict(args.image_path, args.model_path))
