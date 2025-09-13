import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)  # pneumonia / normal

model.load_state_dict(torch.load("pneumonia_classifier_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = "./chest_xray/test/PNEUMONIA/person124_bacteria_590.jpeg"

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

classes = ["Normal", "Pneumonia"]
print(f"Prediction: {classes[pred]}")
