import os

import torch
import torch.nn as nn
from torchvision import transforms

from dataset import train_loader, val_loader
from model import CNN_BN

model = CNN_BN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-5
)

augmentation = transforms.RandomAffine(
    degrees=10,
    translate=(0.08, 0.08)
)

os.makedirs("outputs", exist_ok=True)

best_accuracy = 0.0

for epoch in range(20):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = torch.stack([
            augmentation(image) for image in images
        ])

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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

    if accuracy > best_accuracy:
        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "outputs/best_cnn_combined.pth"
        )

    scheduler.step()

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss / len(train_loader):.4f}, "
        f"Val Accuracy: {accuracy * 100:.2f}%"
    )

print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")