# coding=UTF-8
"""
  3. model

  from torch import nn
  from torchvision import models
  from torch.utils import model_zoo
"""

import os
import torch

from torch.utils.data import Dataset, DataLoader, dataset
from torchvision import models, transforms

transform = transforms.Compose(
  [
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
  ]
)
from torchvision.datasets.mnist import MNIST
train_dataset = MNIST(root="./mnist_data",
                      train=True,
                      transform=transform,
                      target_transform=None,
                      download=False)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10000,
                          shuffle=True)

"""
  TODOEg3.0.0 : torch.nn.Module
"""

def eg_3_0_0():
  from torch import nn
  class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

  model = SimpleModel()
  print("model: {}".format(model))
  for name, param in model.named_parameters():
    print(name, param)

"""
  TODOEg3.0.1 : super().__init__()
"""

def eg_3_0_1():
  from torch import nn
  class SimpleModel(nn.Module):
    def __init__(self):
        # super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

  model = SimpleModel()
  print("model: {}".format(model))

"""
  TODOEg3.1 : __call__  [magic methods]
"""

def eg_3_1():
  from torch import nn
  class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

  model = SimpleModel()
  x = train_dataset[0][0]  # torch.Size([1, 28, 28])
  x = x[None, ...]  # torch.Size([1, 1, 28, 28])
  print(model(x) == model.forward(x))

"""
  TODOEg3.2 : (B, C, H ,W)
"""

def eg_3_2():
  from torch import nn
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
        print("[before flatten] x.shape: {}".format(x.shape))  # torch.Size([1, 5, 28, 28])
        x = self.flatten(x)
        print("[after flatten] x.shape: {}".format(x.shape))  # torch.Size([1, 3920])
        x = self.linear(x)
        x = self.relu(x)
        return x

  model = SimpleModel()
  x = train_dataset[0][0]  # torch.Size([1, 28, 28])
  x = x[None, ...]  # torch.Size([1, 1, 28, 28])
  model(x)

"""
  TODOEg3.3 : torchvision.models
"""

def eg_3_3():
  from torchvision import models

  model_vgg16 = models.vgg16()
  print("model_vgg16: {}".format(model_vgg16))

  model_resnet50 = models.resnet50()
  print("model_resnet50: {}".format(model_resnet50))

"""
  TODOEg3.4.0 : model.state_dict()
"""

def eg_3_4_0():
  from torchvision import models

  model_vgg16 = models.vgg16()
  print("model_vgg16.state_dict(): {}".format(model_vgg16.state_dict()))

"""
  TODOEg3.4.1 : torch.save(model.state_dict(), f)
"""

def eg_3_4_1():
  from torchvision import models

  model_vgg16 = models.vgg16()
  torch.save(model_vgg16.state_dict(), "./vgg16.pth",)

"""
  TODOEg3.4.2 : model.load_state_dict()
"""

def eg_3_4_2():
  from torchvision import models

  model_vgg16 = models.vgg16()
  state_dict = torch.load("./vgg16.pth", map_location="cpu")
  missing_keys, unexpected_keys = model_vgg16.load_state_dict(state_dict, strict=True)
  print("missing_keys: {}".format(missing_keys))
  print("unexpected_keys: {}".format(unexpected_keys))

"""
  TODOEg3.4.3 : strict=False
"""

def eg_3_4_3():
  from torchvision import models

  model_vgg16 = models.vgg16()
  state_dict = torch.load("./vgg16.pth", map_location="cpu")
  for key in list(state_dict.keys()):
    if ".bias" in key:
      del state_dict[key]

  missing_keys, unexpected_keys = model_vgg16.load_state_dict(state_dict, strict=False)
  print("missing_keys: {}".format(missing_keys))
  print("unexpected_keys: {}".format(unexpected_keys))

"""
  TODOEg3.5 : torch.utils.model_zoo
"""
def eg_3_5():
  from torch.utils import model_zoo
  from torchvision import models

  model_alexnet = models.alexnet()
  state_dict = model_zoo.load_url('http://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
  model_alexnet.load_state_dict(state_dict)

if __name__ == "__main__":
  """
  3.0.0 torch.nn.Module
  3.0.1 super().__init__()
  3.1 __call__  [magic methods]
  3.2 (B, C, H ,W)
  3.3 torchvison.models
  3.4.0 model.dict_state()
  3.4.1 torch.save(model.dict_state(), f)
  3.4.2 model.load_state_dict()
  3.4.3 strict=False
  3.5 torch.utils.model_zoo.load_url()
  """

  # eg_3_0_0()
  # eg_3_0_1()
  # eg_3_1()
  # eg_3_2()
  # eg_3_3()
  # eg_3_4_0()
  # eg_3_4_1()
  # eg_3_4_2()
  # eg_3_4_3()
  eg_3_5()

  print("~~~~~~下课~~~~~~")