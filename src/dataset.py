import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader

df = pd.read_csv('data/train.csv')
print(df.shape)
print(df.head())#默认返回前5行
#监督学习中X(features) Y(labels)

x = df.drop(columns=["label"])
y = df["label"]

print(x.shape)
print(y.shape)

print(x.head())
print(y.head())

x_normalized = x / 255.0
print(x_normalized.max().max())
print(x_normalized.min().min())

X_tensor = torch.tensor(x_normalized.to_numpy(), dtype=torch.float32)
y_tensor = torch.tensor(y.to_numpy(), dtype=torch.long)

X_images = X_tensor.reshape(-1,1,28,28)
print(X_images.shape)

dataset = TensorDataset(X_images, y_tensor)
print(len(dataset))

train_dataset, val_dataset = random_split(
    dataset, 
[0.9,0.1],
generator=torch.Generator().manual_seed(42)
)
print(len(train_dataset))
#Dataloader是PyTorch中用于加载数据的模块，它将数据集封装成一个可迭代对象，方便进行批量处理和并行计算。
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)