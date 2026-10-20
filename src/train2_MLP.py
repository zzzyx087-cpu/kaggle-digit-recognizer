import torch
import torch.nn as nn
from dataset import train_loader, val_loader
from model import MLP
import tqdm
torch.manual_seed(42)
# 1. 创建 MLP 模型
model = MLP()

# 2. 损失函数
criterion = nn.CrossEntropyLoss()

# 3. 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练
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

    print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(train_loader):.4f}")
# 5. 验证
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
print(f"MLP Validation Accuracy: {accuracy*100:.2f}%")
torch.save(model.state_dict(), "mlp_model.pth")