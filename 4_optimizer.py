# coding=UTF-8
"""
  4. optimizer

  from torch.nn import opt
"""

import os
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets.mnist import MNIST

transform = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ]
)

train_dataset = MNIST(root="./mnist_data",
                      train=True,
                      transform=transform,
                      target_transform=None,
                      download=False)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10000,
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

model = SimpleModel()


def eg_4_0():
  """
  Eg4.0 : torch.optim
  """
  from torch import optim
  optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.9)
  print("optim.state_dict(): {}".format(optimizer.state_dict()))


def eg_4_1():
  """
  Eg4.1 : params
  """
  from torch import optim
  params = [param for name, param in model.named_parameters() if ".bias" in name]
  optimizer = optim.SGD(params=params, lr=0.0001, momentum=0.9)
  print("optim.state_dict(): {}".format(optimizer.state_dict()))


def eg_4_2():
  """
  Eg4.2 : zero_grad(), step()
  """
  from torch import optim
  from tqdm import tqdm
  optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
  loss_fn = nn.CrossEntropyLoss()

  for epoch in range(2):
    with tqdm(train_loader, desc="EPOCH: {}".format(epoch)) as train_bar:
      for (x, y) in train_bar:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    print("epoch: {},  loss: {:.6f}".format(epoch, loss))


if __name__ == "__main__":
  """
  4.0 torch.optim
  4.1 params
  4.2 zero_grad(), step()
  """

  # eg_4_0()
  # eg_4_1()
  eg_4_2()



  print("~~~~~~下课~~~~~~")