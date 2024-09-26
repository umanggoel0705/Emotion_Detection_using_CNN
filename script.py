import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

###################

data_dir = "data/train"
transform = T.Compose([T.RandomHorizontalFlip(),
                       T.RandomRotation(degrees=25),
                       T.ToTensor(),
                       T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
dataset = datasets.ImageFolder(
    root_dir := data_dir,
    transform = transform
)

####################

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
batch_size = 1024
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# for x,y in train_dl:
#     print("Shape of x:",x.size())
#     print("Shape of y:",y.size())
#     break

device = "cuda"

def train(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        # print(pred.shape)
        loss = loss_fn(pred, y)

        temp.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch%10 == 0:
            loss, current = loss.item(), batch*len(x)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    Train_loss.append(np.mean(temp))

prevAcc = 0.0
def validation(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= batch_num
    Test_loss.append(test_loss)
    correct /= size
    print(f"Test Error:\n Accuracy: {correct*100:>0.1f}%  Avg Loss: {test_loss:>8f} \n")

    global prevAcc
    if prevAcc < correct:
        ########
        torch.save(model.state_dict(), "model.pth")
        ########
        print("Saved Successfully!!\n")
        prevAcc = correct

#######################

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(64*6*6, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 7),
            nn.Softmax()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

#########################

model = NeuralNetwork().to(device)
# print(model)

model1 = NeuralNetwork().to(device)
model1.load_state_dict(torch.load("model.pth"))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)

Train_loss = []
Test_loss = []
epochs = 100
# for t in range(epochs):
#     temp = []
#     print(f"Epoch {t+1} ----------------------\n")
#     train(model1, train_dl, loss_fn, optimizer)
#     validation(model1, val_dl, loss_fn)
# print(f"Done!!, Accuracy: {prevAcc*100:>6f}")

# ##########################

# x = torch.linspace(0, len(Train_loss), len(Train_loss))
# plt.plot(x, Train_loss)
# plt.title("Train Loss")
# plt.show()

# x = torch.linspace(0, len(Test_loss), len(Test_loss))
# plt.plot(x, Test_loss)
# plt.title("Test Loss")
# plt.show()

test_dir = "data/test"
transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                       ])
test_dataset = datasets.ImageFolder(
    root_dir := test_dir,
    transform=transform
)

model1 = NeuralNetwork().to(device)
model1.load_state_dict(torch.load("model.pth"))

test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validation(model1, test_dl, loss_fn)
print("Done!!")