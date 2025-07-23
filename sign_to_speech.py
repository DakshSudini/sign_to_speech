import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"🟢 CUDA is available: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU memory allocated (MB): {torch.cuda.memory_allocated() / 1024**2:.1f}")
    print(f"📊 GPU memory reserved  (MB): {torch.cuda.memory_reserved() / 1024**2:.1f}")
else:
    print("🔴 CUDA is not available. Training will use CPU.")

# ✅ Paths
DATA_DIR = "asl_alphabet/asl_alphabet_train"

MODEL_PATH = "asl_cnn_model.pth"
BATCH_SIZE = 32
EPOCHS = 10

# ✅ Data transforms with augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Dataset and Dataloader
dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
class_names = dataset.classes
print("🔤 Classes:", class_names)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ✅ Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training Loop
print("🔁 Starting training...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"📉 Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

# ✅ Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
