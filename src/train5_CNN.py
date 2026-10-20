import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import train_loader, val_loader
from model import CNN
import os

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:

        # 1. Clear previous gradients
        optimizer.zero_grad()

        # 2. Forward propagation
        outputs = model(images)

        # 3. calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update parameters
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        # Forward propagation
        outputs = model(images)

        # Get predicted classes
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

os.makedirs("outputs", exist_ok=True)
torch.save(
    model.state_dict(),
    "outputs/cnn_baseline.pth"
)

print("Model saved successfully!")