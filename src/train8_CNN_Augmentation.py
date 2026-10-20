import torch
import torch.nn as nn
from torchvision import transforms
import os
from dataset import train_loader, val_loader
from model import CNN

model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

augmentation = transforms.RandomAffine(
    degrees=10,
    translate=(0.08, 0.08)
)

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = augmentation(images)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

os.makedirs("outputs", exist_ok=True)
torch.save(
    model.state_dict(),
    "outputs/cnn_augmentation.pth"
)