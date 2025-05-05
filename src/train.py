import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm # Optional progress bar
import os
import argparse
import json
import time
import copy
from data_utils import get_transforms # 从 data_utils.py 导入


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) # For best PEFT weights saving
    best_adapter_wts = None # Store best LoRA adapter weights separately
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Check if it's a PEFT model to save adapters correctly
    is_peft_model = hasattr(model, 'save_pretrained')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Add tqdm progress bar if desired
            data_iterator = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}") if phase == 'train' else dataloaders[phase]

            for inputs, labels in data_iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # Note: Output might be different for PEFT models depending on task_type, adjust if needed
                    # Assuming standard classification output
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # Optional: Update progress bar description
                        # data_iterator.set_postfix(loss=loss.item())


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                 scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

             # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                 # Save best model based on validation accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if is_peft_model:
                        # For PEFT, we typically save the adapter weights
                        best_adapter_wts = {k: v.cpu().clone() for k, v in model.named_parameters() if "lora_" in k}
                        print(f'Best val Acc: {best_acc:.4f}, storing LoRA adapter weights.')
                    else:
                         # For non-PEFT, save the whole state dict (or relevant parts)
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print(f'Best val Acc: {best_acc:.4f}, storing model state dict.')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

     # Load best model weights/adapters before returning
    if is_peft_model and best_adapter_wts:
        # Load the best adapter weights back into the model
        # This part might need refinement depending on how PEFT handles loading partial state dicts
        # Usually, saving/loading the adapter via save_pretrained/from_pretrained is preferred
        print("Loading best LoRA adapter weights (in-memory, recommend save_pretrained).")
        # model.load_state_dict(best_adapter_wts, strict=False) # Might not work directly
        pass # Prefer saving outside the function using save_pretrained
    elif not is_peft_model:
        model.load_state_dict(best_model_wts)


#     return model, history, best_acc # Return best accuracy as well

def main(args):
    # --- Setup ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    print("Loading data...")
    data_transforms = get_transforms() # 从 data_utils 获取 transforms
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes. Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")

    # 保存 class_names
    class_names_path = os.path.join(args.output_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {class_names_path}")

    # --- Model ---
    print("Loading base model...")
    # TODO: Add flexibility for different base models if needed
    model_base = models.resnet50(weights=None) # Start with no weights, load later
    num_ftrs = model_base.fc.in_features
    model_base.fc = nn.Linear(num_ftrs, num_classes)

    if args.base_model_weights:
        print(f"Loading base weights from: {args.base_model_weights}")
        try:
            model_base.load_state_dict(torch.load(args.base_model_weights, map_location='cpu')) # Load to CPU first
            print("Base weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load base weights: {e}. Starting from scratch or default ImageNet?")
            # Optionally load ImageNet weights here if loading fails
            # model_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # model_base.fc = nn.Linear(model_base.fc.in_features, num_classes)

    # --- LoRA Setup ---
    print("Applying LoRA...")
    # Allow target_modules to be specified as a list of strings in the args
    target_modules_list = args.target_modules.split() if args.target_modules else ["fc"] # Default to fc if none specified
    print(f"Targeting LoRA modules: {target_modules_list}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules_list,
        lora_dropout=0.05,
        bias="none",
        # task_type=TaskType.IMAGE_CLASSIFICATION, # Avoid task_type for torchvision models
        modules_to_save = ["fc"] if "fc" not in target_modules_list else None # Also train fc if not targeted by LoRA? Decide strategy.
    )
    model_lora = get_peft_model(model_base, lora_config)
    model_lora.print_trainable_parameters()
    model_lora = model_lora.to(device)

    # --- Training ---
    print("Setting up optimizer and scheduler...")
    optimizer = optim.Adam(model_lora.parameters(), lr=args.lr)
    # TODO: Add scheduler options via args if needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    # model_trained, history, best_val_acc = train_model( # Use the imported/pasted train_model function
    #     model_lora, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=args.epochs
    # )
    # TODO: Paste or import the train_model function here and uncomment the call above
    # Placeholder print statement until train_model is integrated
    print("TRAINING FUNCTION NEEDS TO BE INTEGRATED HERE")
    best_val_acc = 0 # Placeholder

    # --- Save Final Model (LoRA Adapters) ---
    adapter_save_path = os.path.join(args.output_dir) # Save directly in output dir
    print(f"Saving final LoRA adapters to {adapter_save_path}...")
    # model_trained.save_pretrained(adapter_save_path) # Use the returned trained model
    # Placeholder print statement until train_model is integrated
    print("MODEL SAVING NEEDS TO BE INTEGRATED HERE USING THE RETURNED MODEL")

    print("-" * 30)
    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"LoRA adapters saved in: {adapter_save_path}")
    print(f"Class names saved in: {class_names_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a bird classification model with LoRA.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the ImageFolder dataset (containing train/val).')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save LoRA adapters and logs.')
    parser.add_argument('--base_model_weights', type=str, default=None, help='(Optional) Path to pre-trained weights for the base model (e.g., head-trained weights).')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA.')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA.')
    parser.add_argument('--target_modules', type=str, default="fc", help='Space-separated list of module names to apply LoRA to (e.g., "fc conv1 layer4.0.conv1").')

    args = parser.parse_args()
    main(args)
