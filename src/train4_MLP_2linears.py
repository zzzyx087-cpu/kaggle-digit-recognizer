import torch
import torch.nn as nn
from dataset import train_loader, val_loader
from model import MLPDeep

torch.manual_seed(42)

model = MLPDeep()
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0.0
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/100, Loss: {total_loss / len(train_loader):.4f}")

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
print(f"MLPDeep Validation Accuracy: {accuracy * 100:.2f}%")
torch.save(model.state_dict(), "mlp_deep_model.pth")
