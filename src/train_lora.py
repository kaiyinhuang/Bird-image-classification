from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from data_loader import load_bird_dataset, extract_label, get_transforms, load_image
import numpy as np
import torch
import evaluate

# 加载数据集（同全参数微调）
dataset = load_bird_dataset().map(extract_label)
label_list = dataset["train"].unique("label")
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 应用预处理（同全参数微调）
dataset["train"] = dataset["train"].map(
    lambda x: load_image(x, get_transforms("train")),
    batched=False
)
dataset["validation"] = dataset["validation"].map(
    lambda x: load_image(x, get_transforms("val")),
    batched=False
)

# 加载基础模型
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 添加 LoRA 适配器
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],  # ViT 注意力层的 query 和 value 矩阵
    lora_dropout=0.05,
    modules_to_save=["classifier"],      # 分类层保持全参数训练
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 输出示例: trainable params: 0.5M / 86M (0.58%)

# 定义训练参数（学习率更大）
training_args = TrainingArguments(
    output_dir="./outputs/lora",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    learning_rate=1e-4,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# 训练与保存
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=lambda p: {"accuracy": np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)}
)
trainer.train()
model.save_pretrained("./outputs/lora")
