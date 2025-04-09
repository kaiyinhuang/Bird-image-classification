import os
from datasets import load_dataset, DatasetDict
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomRotation
from PIL import Image

def load_bird_dataset() -> DatasetDict:
    """从 Hugging Face 加载 gjuggler/bird-data 数据集"""
    dataset = load_dataset("gjuggler/bird-data")
    return dataset

def extract_label(example: dict) -> dict:
    """从图像路径中提取类别标签（例如 'Abert's Towhee'）"""
    path = example["image_file_path"]
    class_name = path.split("/")[-2].replace("zip://", "")
    return {"label": class_name}

def get_transforms(mode: str = "train") -> Compose:
    """定义图像预处理流水线"""
    if mode == "train":
        return Compose([
            Resize(256),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return Compose([
            Resize(256),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_image(example: dict, transform: Compose) -> dict:
    """加载图像并应用预处理"""
    try:
        image_path = example["image_file_path"].replace("zip://", "")
        image = Image.open(image_path).convert("RGB")
        example["image"] = transform(image)
        return example
    except Exception as e:
        print(f"加载图像失败: {example['image_file_path']}, 错误: {e}")
        return None
