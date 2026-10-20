import torch
import torch.nn as nn
from dataset import train_loader

model = nn.Linear(784, 10)

print(model)
print(model.weight.shape)
print(model.bias.shape)

images = torch.randn(64,784) #形状为64 784的随机tensor
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("First image output:", outputs[0])

images, labels = next(iter(train_loader))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

images = images.reshape(images.shape[0], -1)
print(images.shape)

outputs = model(images)
print(outputs.shape)
print(outputs[0])
print("True label:", labels[0].item())
class LogisticRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)#784是输入维度，128是输出维度
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLP256(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLPDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(64 * 5 * 5, 10)

    def forward(self, x):
        #first block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        #second block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
class CNN_BN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 10)
        )

    def forward(self, x):
        return self.network(x)
        
if __name__ == "__main__":

    model = CNN()

    x = torch.randn(64, 1, 28, 28)

    output = model(x)

    print("Output shape:", output.shape)