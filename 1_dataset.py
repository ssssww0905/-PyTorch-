# coding=UTF-8
"""
  1. Dataset

  from torch.utils.data import Dataset
  from torchvision.datasets.mnist import MNIST
  from torchvision.datasets.voc import VOCSegmentation, VOCDetection
  from torchvision.datasets import ImageFolder
"""

import os
import torch

from torch.utils.data import Dataset
from torch.utils.data import Sampler


def eg_1_1():
  """
  Eg1.1 : __getitem__, __len__
  """
  x = torch.linspace(-1, 1, 10)
  y = x**2

  class SimpleDataset(Dataset):
    def __init__(self, x, y):
      super().__init__()
      self.x = x
      self.y = y

    def __getitem__(self, index):
      return {"x":self.x[index], "y":self.y[index]}

    def __len__(self):
      return len(self.x)

  simpledataset = SimpleDataset(x, y)
  index = 0
  # __getitem__
  print("simpledataset.__getitem__({}): {}".format(index, simpledataset.__getitem__(index)))
  print("simpledataset[{}]: {}".format(index, simpledataset[index]))
  # __len__
  print("simpledataset.__len__(): {}".format(simpledataset.__len__()))
  print("len(simpledataset): {}".format(len(simpledataset)))


def eg_1_2_0():
  """
  Eg1.2.0 : MNIST
  """
  from torchvision.datasets.mnist import MNIST
  train_dataset = MNIST(root="./mnist_data",
                        train=True,
                        transform=None,
                        download=True)

  print("type(train_dataset): {}".format(type(train_dataset)))  # <class 'torchvision.datasets.mnist.MNIST'>
  index = 0
  print("train_dataset[{}]: {}".format(index, train_dataset[index]))  # (PIL.Image.Image, 5)
  print("len(train_dataset): {}".format(len(train_dataset)))

  import matplotlib.pyplot as plt
  plt.imshow(train_dataset[index][0], cmap ='gray')


def eg_1_2_1():
  """
  Eg1.2.1 : transforms
  """
  from torchvision.datasets.mnist import MNIST
  from torchvision import transforms

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

  index = 0
  print("type(train_dataset[{}]): {}".format(index, type(train_dataset[index])))  # <class 'tuple'>
  print("type(train_dataset[{}][0]): {}".format(index, type(train_dataset[index][0])))  # <class 'torch.Tensor'>
  print("train_dataset[{}][0].shape: {}".format(index, train_dataset[index][0].shape))  # torch.Size([1, 28, 28])
  print("type(train_dataset[{}][1]): {}".format(index, type(train_dataset[index][1])))  # <class 'int'>


def eg_1_3():
  """
  Eg1.3 : VOCSegmentation, VOCDetection
  """
  from torchvision.datasets.voc import VOCSegmentation, VOCDetection

  segmentation_dataset = VOCSegmentation(root="./voc_data",
                                        image_set="train",
                                        transform=None,
                                        download=False)
  detection_dataset = VOCDetection(root="./voc_data",
                                  image_set="train",
                                  transform=None,
                                  download=False)

  index = 0
  print("type(segmentation_dataset[{}]): {}".format(index, type(segmentation_dataset[index])))  # <class 'tuple'>
  print("type(segmentation_dataset[{}][0]): {}".format(index, type(segmentation_dataset[index][0])))  # <class 'PIL.Image.Image'>
  print("type(segmentation_dataset[{}][1]): {}".format(index, type(segmentation_dataset[index][1])))  # <class 'PIL.PngImagePlugin.PngImageFile'>

  print("type(detection_dataset[{}]): {}".format(index, type(detection_dataset[index])))  # <class 'tuple'>
  print("type(detection_dataset[{}][0]): {}".format(index, type(detection_dataset[index][0])))  # <class 'PIL.Image.Image'>
  print("type(detection_dataset[{}][1]): {}".format(index, type(detection_dataset[index][1])))  # <class 'dict'>


def eg_1_4_0():
  """
  Eg1.4.0 : ImageFolder
  """
  from torchvision.datasets import ImageFolder
  from torchvision import transforms

  transform = transforms.Compose(
    [
      transforms.RandomResizedCrop(size=(224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
  )
  train_dataset = ImageFolder(root=os.path.join("./flower_data", "train"), transform=transform)

  index = 0
  print("type(train_dataset[{}]): {}".format(index, type(train_dataset[index])))  # <class 'tuple'>
  print("type(train_dataset[{}][0]): {}".format(index, type(train_dataset[index][0])))  # <class 'torch.Tensor'>
  print("train_dataset[{}][0].shape: {}".format(index, train_dataset[index][0].shape))  # torch.Size([3, 224, 224])
  print("type(train_dataset[{}][1]): {}".format(index, type(train_dataset[index][1])))  # <class 'int'>


def eg_1_4_1():
  """
  Eg1.4.1 : classes, class_to_idx
  """
  from torchvision.datasets import ImageFolder
  from torchvision import transforms

  transform = transforms.Compose(
    [
      transforms.RandomResizedCrop(size=(224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
  )
  train_dataset = ImageFolder(root=os.path.join("./flower_data", "train"), transform=transform)

  print("train_dataset.classes: {}".format(train_dataset.classes))  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
  print("train_dataset.class_to_idx: {}".format(train_dataset.class_to_idx))  # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


if __name__ == "__main__":
  """
  1.0 torch.utils.data.Dataset
  1.1 __getitem__, __len__  [magic methods]
  1.2.0 MNIST
  1.2.1 transforms
  1.3 VOCSegmentation, VOCDetection
  1.4.0 ImageFolder
  1.4.1 classes, class_to_idx
  """

  eg_1_1()
  # eg_1_2_0()
  # eg_1_2_1()
  # eg_1_3()
  # eg_1_4_0()
  # eg_1_4_1()

  # print("~~~~~~下课~~~~~~")



