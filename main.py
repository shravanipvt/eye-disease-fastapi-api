from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://www.dropbox.com/scl/fi/r4exghzf0knsr3tc22k3r/model.pth?rlkey=jkfc535ez7kljthdor1yjw5u4&st=35cx4i3q&dl=1"
MODEL_PATH = "models/model.pth"

def download_file_direct(url, destination):
    if os.path.exists(destination):
        print("Model already exists.")
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("Model downloaded successfully.")


class EfficientNetB4Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(pretrained=False)
        self.head = nn.Linear(1000, 8)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB4Classifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


download_file_direct(MODEL_URL, MODEL_PATH)
MODEL, DEVICE = load_model()

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

CLASSES = {
    0: "Normal",
    1: "Diabetes",
    2: "Glaucoma",
    3: "Cataract",
    4: "Age-related Macular Degeneration",
    5: "Hypertensive Retinopathy",
    6: "Myopia",
    7: "Unclassified case"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
