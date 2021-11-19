# coding=UTF-8
"""
  5. train

"""
import os
from datetime import datetime
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

transform = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ]
)
#------------------dataset------------------#
train_dataset = MNIST(root="./mnist_data",
                      train=True,
                      transform=transform,
                      target_transform=None,
                      download=False)
#-----------------dataloader----------------#
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=100,
                          shuffle=True)

class SimpleModel(nn.Module):
  def __init__(self):
      super(SimpleModel, self).__init__()
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
      self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(1, 1))
      self.relu = nn.ReLU(inplace=True)
      self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
      self.linear = nn.Linear(in_features=5*28*28, out_features=10, bias=False)

  def forward(self, x):
      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.flatten(x)
      x = self.linear(x)
      x = self.relu(x)
      return x
#-------------------model------------------#
model = SimpleModel()
model.load_state_dict(torch.load("./model_2021_11_19.pth"))
#-----------------optimizer----------------#
optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
  with tqdm(train_loader, desc="EPOCH: {}".format(epoch)) as train_bar:
    for (x, y) in train_bar:
      optimizer.zero_grad()
      loss = loss_fn(model(x), y)
      loss.backward()
      optimizer.step()
  print("epoch: {},  loss: {:.6f}".format(epoch, loss))

time = str(datetime.now()).split(" ")[0].replace("-", "_")
torch.save(model.state_dict(), "model_{}.pth".format(time))

print("~~~~~~下课~~~~~~")
