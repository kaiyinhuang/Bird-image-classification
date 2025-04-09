from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from data_loader import load_bird_dataset, extract_label, get_transforms, load_image
import numpy as np
import torch
import evaluate

# 加载数据集并预处理
dataset = load_bird_dataset().map(extract_label)
label_list = dataset["train"].unique("label")
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 应用预处理
dataset["train"] = dataset["train"].map(
    lambda x: load_image(x, get_transforms("train")),
    batched=False
)
dataset["validation"] = dataset["validation"].map(
    lambda x: load_image(x, get_transforms("val")),
    batched=False
)

# 加载模型
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./outputs/full_ft",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=3e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb"
)

# 定义评估指标
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()
model.save_pretrained("./outputs/full_ft")
