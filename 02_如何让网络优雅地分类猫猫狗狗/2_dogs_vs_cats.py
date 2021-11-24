# coding=UTF-8
"""
2. Dogs vs. Cats
    https://www.kaggle.com/c/dogs-vs-cats
"""
import argparse
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.serialization import save
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=False)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self, name, fmt=':.6f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} val: {val' + self.fmt + '} avg: {avg' + self.fmt + '} sum: {sum' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def train(train_loader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    train_loss_record = AverageMeter(name="train_loss_record")
    train_acc1_record = AverageMeter(name="train_acc1_record")

    with tqdm(train_loader, desc="TRAIN EPOCH: {}".format(epoch)) as train_bar:
        for (data, target) in train_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target, topk=(1,))
            train_acc1_record.update(acc1[0].item(), data.size(0))
            train_loss_record.update(loss.item(), data.size(0))

    writer.add_scalar("train_loss", train_loss_record.avg, epoch)
    writer.add_scalar("train_acc1", train_acc1_record.avg, epoch)


def validate(val_loader, model, loss_fn, epoch, writer, save_dict):
    model.eval()

    val_loss_record = AverageMeter(name="val_loss_record")
    val_acc1_record = AverageMeter(name="val_acc1_record")
    with torch.no_grad():
        with tqdm(val_loader, desc="VALID EPOCH: {}".format(epoch)) as val_bar:
            for (data, target) in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)

                acc1 = accuracy(output, target, topk=(1,))
                val_acc1_record.update(acc1[0].item(), data.size(0))
                val_loss_record.update(loss.item(), data.size(0))

    writer.add_scalar("val_loss", val_loss_record.avg, epoch)
    writer.add_scalar("val_acc1", val_acc1_record.avg, epoch)

    if val_acc1_record.avg > save_dict["max"]:
        save_dict.update({"max":val_acc1_record.avg, "epoch":epoch, "state_dict":model.state_dict()})


def main():
    # argparse
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=625,
                        help='input batch size for training (default: 625)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = ImageFolder(root="./train", transform=transform)
    val_dataset = ImageFolder(root="./val", transform=transform)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # model
    model = models.AlexNet(num_classes=2)
    state_dict = torch.utils.model_zoo.load_url('http://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    for key in list(state_dict.keys()):
        if "classifier.6" in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    # train
    save_dict = {"max":0, "epoch":0, "state_dict":model.state_dict()}
    with SummaryWriter("dog_vs_cat") as writer:
        for epoch in range(3):
        # for epoch in range(args.epochs):
            train(train_loader, model, loss_fn, optimizer, epoch, writer)
            validate(val_loader, model, loss_fn, epoch, writer, save_dict)

    time = str(datetime.now()).split(" ")[0].replace("-", "_")
    torch.save(save_dict["state_dict"], "epoch_{}_{}_{}.pth".format(save_dict["epoch"], device, time))

if __name__ == "__main__":
    # main()

    print("~~~~~~下课~~~~~~")

