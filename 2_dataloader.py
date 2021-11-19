# coding=UTF-8
"""
  2. DataLoader

  from torch.utils.data import DataLoader
"""

import os
import torch

from torch.utils.data import Dataset
from torch.utils.data._utils import collate
from torchvision import transforms

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

"""
  TODOEg2.1 : __iter__
"""

def eg_2_1():
  from torch.utils.data import DataLoader
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False)

  print("type(train_loader): {}".format(type(train_loader)))  # <class 'torch.utils.data.dataloader.DataLoader'>
  for batch in train_loader:
    print("type(batch): {}".format(type(batch)))  # <class 'list'>
    print("len(batch): {}".format(len(batch)))  # 2
    print("type(batch[0]): {}".format(type(batch[0])))  # <class 'torch.Tensor'>
    print("type(batch[1]): {}".format(type(batch[0])))  # <class 'torch.Tensor'>
    print("batch[0].shape: {}".format(batch[0].shape))  # torch.Size([10000, 1, 28, 28])
    print("batch[1].shape: {}".format(batch[1].shape))  # torch.Size([10000])
    break

"""
  TODOEg2.2 : __len__
"""

def eg_2_2():
  from torch.utils.data import DataLoader
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False)

  print("len(train_loader): {}".format(len(train_loader)))  # 6
  print("len(train_loader.dataset): {}".format(len(train_loader.dataset)))  # 60000

"""
  TODOEg2.3.0 : enumerate
"""

def eg_2_3_0():
  from torch.utils.data import DataLoader
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False)

  for batch, (x, y) in enumerate(train_loader):
    print("batch: {}, type(x): {}, type(y): {}".format(batch, type(x), type(y)))
    # batch: 0, type(x): <class 'torch.Tensor'>, type(y): <class 'torch.Tensor'>
    print("batch: {}, x.shape: {}, y.shape: {}".format(batch, x.shape, y.shape))
    # batch: 0, x.shape: torch.Size([10000, 1, 28, 28]), y.shape: torch.Size([10000])
    break

"""
  TODOEg2.3.1 : tqdm
"""

def eg_2_3_1():
  from torch.utils.data import DataLoader
  from tqdm import tqdm
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False)

  with tqdm(train_loader, desc="TRAINING") as train_bar:
    for (x, y) in train_bar:
      pass

"""
  TODOEg2.4 : tqdm
"""

def eg_2_4():
  def collate_fn(batch):
    print("type(batch): {}, len(batch): {}".format(type(batch), len(batch)))  # <class 'list'>, 10000
    x = [i[0] for i in batch]
    y = [i[1] for i in batch]
    x = torch.cat(x)[:,None,...]
    y = torch.Tensor(y)
    return {"x":x, "y":y}

  from torch.utils.data import DataLoader
  from tqdm import tqdm
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=10000,
                            shuffle=False,
                            collate_fn=collate_fn)

  for batch in train_loader:
    print("type(batch): {}".format(type(batch)))  # <class 'dict'>
    print("type(batch[\"x\"]): {}".format(type(batch["x"])))  # <class 'torch.Tensor'>
    print("type(batch[\"y\"]): {}".format(type(batch["y"])))  # <class 'torch.Tensor'>
    print("batch[\"x\"].shape: {}".format(batch["x"].shape))  # torch.Size([10000, 1, 28, 28])
    print("batch[\"y\"].shape: {}".format(batch["y"].shape))  # torch.Size([10000])
    break

if __name__ == "__main__":
  """
  2.0 torch.utils.data.Dataloader
  2.1 __iter__  [magic methods]
  2.2 __len__  [magic methods]
  2.3.0 enumerate
  2.3.1 tqdm
  2.4 collate_fn
  """

  # eg_2_1()
  # eg_2_2()
  # eg_2_3_0()
  # eg_2_3_1()
  # eg_2_4()

  print("~~~~~~下课~~~~~~")