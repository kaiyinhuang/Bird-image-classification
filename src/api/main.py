# Cell: API Code (main.py content)

# --- Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from torchvision import models
from peft import PeftModel, PeftConfig
from PIL import Image
import io
import json
import os
import uvicorn # Need uvicorn to run the app
import nest_asyncio # Needed for running uvicorn in Colab/Jupyter

# Apply nest_asyncio patch
nest_asyncio.apply()

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200
# --- !!! UPDATE PATHS FOR COLAB !!! ---
CLASS_NAMES_PATH = '/content/bird_class_names.json' # Example path in Colab
LORA_ADAPTER_LOAD_PATH = '/content/bird_classifier_lora_adapters' # Example path in Colab
BASE_MODEL_LOAD_PRETRAINED = True # Use torchvision pre-trained weights

# --- Load Class Names ---
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Class names count mismatch: {len(class_names)} vs {NUM_CLASSES}")
    print(f"Class names loaded from {CLASS_NAMES_PATH}")
except Exception as e:
    print(f"ERROR loading class names: {e}")
    class_names = [f'class_{i}' for i in range(NUM_CLASSES)]

# --- Define Image Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Model Function ---
def load_model():
    print("Loading model...")
    model_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if BASE_MODEL_LOAD_PRETRAINED else None)
    num_ftrs = model_base.fc.in_features
    model_base.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    print("Base model loaded and FC layer modified.")

    try:
        model_lora = PeftModel.from_pretrained(model_base, LORA_ADAPTER_LOAD_PATH)
        model_lora.eval()
        model_lora = model_lora.to(DEVICE)
        print(f"LoRA adapters loaded from {LORA_ADAPTER_LOAD_PATH}. Model ready on {DEVICE}.")
        return model_lora
    except Exception as e:
        print(f"FATAL ERROR loading LoRA adapters: {e}")
        raise e

model = load_model() # Load the model when the cell runs

# --- Create FastAPI App ---
app = FastAPI()

# --- Add CORS Middleware (important for cross-origin requests) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Prediction Endpoint ---
@app.post("/predict/")
async def predict_bird(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        print("Image preprocessed.")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    with torch.no_grad():
        try:
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class_name = class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            print(f"Prediction: {predicted_class_name}, Confidence: {confidence_score:.4f}")
        except Exception as e:
             print(f"Error during model inference: {e}")
             raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    return {
        "predicted_bird": predicted_class_name,
        "confidence": confidence_score
    }

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Bird Classification API is running via Colab!"}

print("FastAPI app defined.")
# Note: We don't run uvicorn directly here yet. We'll use pyngrok to manage it.
