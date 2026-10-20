import torch
import torch.nn as nn
from dataset import train_loader, val_loader
from model import LogisticRegression, MLP


model = LogisticRegression()
criterion = nn.CrossEntropyLoss() #损失函数对象

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 取出一个 Batch

images, labels = next(iter(train_loader))

# 1. 清空上一轮梯度
optimizer.zero_grad()

# 2. 前向传播
outputs = model(images)

# 3. 计算 Loss
loss = criterion(outputs, labels)
print("Loss before update:", loss.item())

# 4. 反向传播
loss.backward()

# 5. 更新参数
optimizer.step()

# 6.训练模型
epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        
        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# 切换到评估模式
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        for i in range(10):
            print(f"Image {i}: Predicted = {predictions[i].item()}, Actual = {labels[i].item()}")
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")