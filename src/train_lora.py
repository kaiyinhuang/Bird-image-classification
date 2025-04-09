from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import yaml
from .data_preprocessing import load_bird_dataset, get_transforms

def main(config_path="configs/lora.yaml"):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 加载数据集和预处理
    dataset = load_bird_dataset()
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    
    # 加载基础模型（冻结所有参数）
    processor = ViTImageProcessor.from_pretrained(config["model_name"])
    model = ViTForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=len(dataset["train"].unique("label")),
        ignore_mismatched_sizes=True
    )
    
    # 添加 LoRA 适配器
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        modules_to_save=config["lora_config"]["modules_to_save"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数占比
    
    # 定义 Trainer
    training_args = TrainingArguments(**config["training_args"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=lambda p: {
            "accuracy": np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)
        }
    )
    
    # 训练与保存
    trainer.train()
    model.save_pretrained(config["training_args"]["output_dir"])

if __name__ == "__main__":
    main()
