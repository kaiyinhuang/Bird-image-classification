from torchvision import transforms
import json

# 标准 ImageNet 均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(input_size=224):
    """Returns a dictionary of training and validation transforms."""
    return {
        'train': transforms.Compose([
            transforms.Resize(256), # 先调整大小
            transforms.RandomCrop(input_size), # 再随机裁剪
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    }

def load_class_names(filepath):
    """Loads class names from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names from {filepath}: {e}")
        return None
