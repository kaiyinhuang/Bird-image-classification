from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from .data_preprocessing import load_bird_dataset, get_transforms
import evaluate
import numpy as np

def main(config_path="configs/full_ft.yaml"):
    # 加载配置
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 数据加载与预处理
    dataset = load_bird_dataset()
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    
    # 模型加载
    processor = ViTImageProcessor.from_pretrained(config["model_name"])
    model = ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=len(dataset["train"].unique("label")),
        ignore_mismatched_sizes=True
    )
    
    # 定义 Trainer
    training_args = TrainingArguments(**config["training_args"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=lambda p: {"accuracy": np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)}
    )
    
    # 训练与保存
    trainer.train()
    model.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()
