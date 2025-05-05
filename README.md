Bird Image Recognition (ResNet50 + LoRA on CUB-200-2011)

This project aims to train a bird image classifier on the CUB-200-2011 dataset using the pre-trained ResNet50 model combined with the LoRA (Low-Rank Adaptation) fine-tuning technique.
Features
Built with PyTorch and torchvision.
Based on the ResNet50 model pre-trained on ImageNet.
Utilizes the parameter-efficient LoRA fine-tuning technique.
Trained on the CUB-200-2011 dataset (200 bird species).
Achieves approximately 70% validation accuracy.
Includes training and inference scripts, as well as instructions for uploading the model to the Hugging Face Hub.
Environment Setup
Clone the Repository:
git clone https://github.com/your_username/your-bird-project.git
cd your-bird-project

Create a Virtual Environment (Recommended):
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

Install Dependencies:
pip install -r requirements.txt

Dataset
This project uses the CUB-200-2011 dataset.
Download: You can download CUB_200_2011.tgz from the official link.
Preparation:
After downloading, extract the file (tar -xzf CUB_200_2011.tgz).
Run the data organization script (or use the logic in the Colab Notebook) to convert it into the ImageFolder format (including train/ and val/ subdirectories, where val/ corresponds to the original test/ split). The structure is as follows:
<dataset_root>/
├── train/
│   ├── 001.Black_footed_Albatross/
│   │   └── ... (images)
│   └── ... (other classes)
└── val/
    ├── 001.Black_footed_Albatross/
    │   └── ... (images)
    └── ... (other classes)

When training, you need to specify the <dataset_root> path through the --data_dir parameter.
Usage
Training
You can use the provided Colab Notebook (notebooks/bird_classification_tuning.ipynb) for interactive training and experimentation, or use the train.py script.
python scripts/train.py \
    --data_dir /path/to/cub_imagefolder_data \
    --output_dir ./output \
    --base_model_weights /path/to/bird_classifier_resnet50_head_trained_10epochs.pth \ # (Optional) Start from the weights of the pre-trained head
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules fc conv1 layer4.0.conv1 \ # (Example) Specify the names of LoRA target modules
    --epochs 15 \
    --lr 1e-4 \
    --batch_size 32

