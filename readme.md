# Bird Image Recognition (ResNet50 + LoRA on CUB-200-2011)

This project aims to train a bird image classifier on the CUB-200-2011 dataset using the pre-trained ResNet50 model combined with the LoRA (Low-Rank Adaptation) fine-tuning technique.

## Features

Built with PyTorch and torchvision.

Based on the ResNet50 model pre-trained on ImageNet.

Utilizes the parameter-efficient LoRA fine-tuning technique.

Trained on the CUB-200-2011 dataset (200 bird species).

Achieves approximately 70% validation accuracy.

Includes training and inference scripts, as well as instructions for uploading the model to the Hugging Face Hub.

## Environment Setup

**Clone the Repository:**



```
git clone https://github.com/your\_username/your-bird-project.git

cd your-bird-project
```

**Create a Virtual Environment (Recommended):**



```
python -m venv venv

source venv/bin/activate  # Linux/macOS

\# venv\Scripts\activate  # Windows
```

**Install Dependencies:**



```
pip install -r requirements.txt
```

## Dataset

This project uses the [CUB-2](https://data.caltech.edu/records/65de6-vp158)[00-20](https://data.caltech.edu/records/65de6-vp158)[11](https://data.caltech.edu/records/65de6-vp158) dataset.

**Download:** You can download `CUB_200_2011.tgz` from the official link.

**Preparation:**

After downloading, extract the file (`tar -xzf CUB_200_2011.tgz`).

Run the data organization script (or use the logic in the Colab Notebook) to convert it into the `ImageFolder` format (including `train/` and `val/` subdirectories, where `val/` corresponds to the original `test/` split). The structure is as follows:



```
\<dataset\_root>/

├── train/

│   ├── 001.Black\_footed\_Albatross/

│   │   └── ... (images)

│   └── ... (other classes)

└── val/

&#x20;   ├── 001.Black\_footed\_Albatross/

&#x20;   │   └── ... (images)

&#x20;   └── ... (other classes)
```

When training, you need to specify the `<dataset_root>` path through the `--data_dir` parameter.

## Usage

### Training

You can use the provided Colab Notebook (`notebooks/bird_classification_tuning.ipynb`) for interactive training and experimentation, or use the `train.py` script.



```
python scripts/train.py \\

&#x20;   \--data\_dir /path/to/cub\_imagefolder\_data \\

&#x20;   \--output\_dir ./output \\

&#x20;   \--base\_model\_weights /path/to/bird\_classifier\_resnet50\_head\_trained\_10epochs.pth \ # (Optional) Start from the weights of the pre-trained head

&#x20;   \--lora\_r 8 \\

&#x20;   \--lora\_alpha 16 \\

&#x20;   \--target\_modules fc conv1 layer4.0.conv1 \ # (Example) Specify the names of LoRA target modules

&#x20;   \--epochs 15 \\

&#x20;   \--lr 1e-4 \\

&#x20;   \--batch\_size 32
```




The script will load the base model and the LoRA adapter, preprocess the images and make predictions, and output the most likely bird name and the confidence level.

### Model Details

**Base Model**: torchvision.models.resnet50 (pre-trained on ImageNet)

**Fine-tuning Strategy**:

(Optional Initial Stage) Only train the classification head (FC layer) to adapt to the CUB dataset.

On this basis, apply LoRA for parameter-efficient fine-tuning.

**LoRA Configuration (Example)**:

r (Rank): 8

lora\_alpha: 16

target\_modules: \["fc", "conv1", "layer4.0.conv1"] (See the training script/configuration for the specific modules used)

lora\_dropout: 0.05

**Final Model**: The trained LoRA adapter can be found on the Hugging Face Hub:

https://huggingface.co/mibo222/bird_classifier_lora/tree/main/bird_classifier_lora_adapters

### Performance

Achieved approximately 70% Top-1 accuracy on the CUB-200-2011 validation set (i.e., the original test set).

(You may consider inserting the training curve graph here)

### Future Work / TODO

Try more powerful base models (such as EfficientNetV2, ViT).

Experiment with more complex LoRA configurations or fine-tuning strategies.

Implement more comprehensive evaluation metrics (Precision, Recall, F1).

Develop a simple Web API (using Flask/FastAPI) for deployment.

Perform model quantization to optimize the inference speed.

### License

This project is licensed under the MIT license.
