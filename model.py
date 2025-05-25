import torch
import torchvision.transforms as transforms
from PIL import Image
import requests

# Load pretrained ResNet18 model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Dummy image
img_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
image = Image.open(requests.get(img_url, stream=True).raw)

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax().item()

print(f"Predicted class index: {pred_class}")
