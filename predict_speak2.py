import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from playsound import playsound
import os
import time

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print("🟢 CUDA is available:", torch.cuda.get_device_name(0))

# Load class names
CLASS_DIR = "asl_alphabet/asl_alphabet_train"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root=CLASS_DIR, transform=transform)
class_names = dataset.classes
print("🧾 Loaded classes:", class_names)

# Load trained model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("asl_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# Init prediction logic
last_prediction = ""
same_count = 0
last_spoken_time = 0
cooldown_seconds = 2  # Minimum seconds between speaking

# Start webcam
cap = cv2.VideoCapture(0)
print("📸 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]

    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)

    # Normalize predictions for comparison
    normalized_prediction = prediction.strip().lower()
    normalized_last = last_prediction.strip().lower()

    if normalized_prediction == normalized_last:
        same_count += 1
    else:
        same_count = 0

    print(f"🗣️ Predicted: {prediction}. Same_count: {same_count}")

    current_time = time.time()
    if same_count == 10 and prediction.lower() != "nothing":
        if current_time - last_spoken_time > cooldown_seconds:
            print(f"🔊 Speaking: {prediction}")
            try:
                tts = gTTS(text=prediction, lang='en')
                temp_mp3 = "temp_tts.mp3"
                tts.save(temp_mp3)
                playsound(temp_mp3)
                os.remove(temp_mp3)
            except Exception as e:
                print(f"❌ TTS failed: {e}")
            last_spoken_time = current_time
            same_count = 0

    last_prediction = prediction  # Save original for speaking

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
