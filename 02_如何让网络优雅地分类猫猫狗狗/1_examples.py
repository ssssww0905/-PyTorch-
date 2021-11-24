# coding=UTF-8
"""
1. Examples
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    https://github.com/pytorch/examples/blob/master/imagenet/main.py

    import argparse
    from torch.utils.tensorboard import SummaryWriter
        writer.add_scalar()
        writer.add_image()
        writer.add_images()
        writer.add_graph()
        tensorboard --logdir=log
    class AverageMeter(object)
        __dict__, __str__
    def accuracy(output, target, topk=(1, ))
        https://pytorch.org/docs/stable/generated/torchtopk.html#torch.topk
    def train(train_loader, model, loss_fn, optimizer, args)
    def validate(val_loader, model, loss_fn, args)
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.module import T
import torch.optim as optim
from torch.utils.tensorboard import writer
import torchvision
from  torchvision import models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def eg_1():
    """
    import argparse  https://docs.python.org/3/library/argparse.html
    """
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=625,
                        help='batch size for training (default: 625)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    print("args: {}".format(args))


def eg_2_1():
    """
    writer.add_scalar()  https://pytorch.org/docs/stable/tensorboard.html?highlight=tensorboard
    """
    from torch.utils.tensorboard import SummaryWriter

    x = []
    y = []
    with open("./log.txt", "w") as f:
        for i in range(10):
            x.append(i)
            y.append(i**2)
            f.write("{}^2 = {}\n".format(i, i**2))
    plt.plot(x, y)
    plt.show()

    with SummaryWriter("./log") as writer:
        for i in range(10):
            writer.add_scalar(tag="i**2", scalar_value=i**2, global_step=i)



def eg_2_2():
    """
    writer.add_image()
    """
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter("./log") as writer:
        cat_image = Image.open("./test/cat.jpg")
        dog_image = Image.open("./test/dog.jpg")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        cat_tensor = transform(cat_image)
        dog_tensor = transform(dog_image)

        img = torchvision.utils.make_grid([cat_tensor, dog_tensor])
        for i in range(10):
            writer.add_image(tag="test_image",
                             img_tensor=img,
                             global_step=i,
                             dataformats="CHW")


def eg_2_3():
    """
    writer.add_images()
    """
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter("./log") as writer:
        cat_image = Image.open("./test/cat.jpg")
        dog_image = Image.open("./test/dog.jpg")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        cat_tensor = transform(cat_image)[None, ...]
        dog_tensor = transform(dog_image)[None, ...]

        imgs = torch.cat([cat_tensor, dog_tensor])  # torch.Size([2, 3, 224, 224])
        for i in range(10):
            writer.add_images("test_images", imgs, global_step=i, dataformats="NCHW")


def eg_2_4():
    """
    writer.add_graph()
    """
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter("./log") as writer:
        cat_image = Image.open("./test/cat.jpg")
        dog_image = Image.open("./test/dog.jpg")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        cat_tensor = transform(cat_image)[None, ...]
        dog_tensor = transform(dog_image)[None, ...]
        imgs = torch.cat([cat_tensor, dog_tensor])

        model = models.AlexNet(num_classes=2)
        for i in range(1):
            writer.add_graph(model, input_to_model=imgs)


def eg_3():
    """
    class AverageMeter(object)
        __dict__, __str__
    """
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
            self.sum += val * n  # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} val: {val' + self.fmt + '} avg: {avg' + self.fmt + '} sum: {sum' + self.fmt + '}'
            print(fmtstr)
            return fmtstr.format(**self.__dict__)

    print("AverageMeter.__dict__: {}".format(AverageMeter.__dict__))
    a = AverageMeter(name="whatever")
    print("\na.__dict__: {}".format(a.__dict__))
    for i in range(10):
        a.update(i)
        print(a)


def eg_4():
    """
    def accuracy(output, target, topk=(1, ))
        https://pytorch.org/docs/stable/generated/torch.topk.html#torch.topk
    """
    def accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            print("pred:\n {}".format(pred))

            print("traget: {}".format(target))
            print("target.view(1, -1).expand_as(pred):\n {}"\
                .format(target.view(1, -1).expand_as(pred)))
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            print("correct:\n {}".format(correct))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    output = torch.Tensor(
        [
            [10, 20, 30],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
            [0.2, 0.1, 0.7]
        ]
    )
    target = torch.Tensor([2, 2, 2, 1])
    print(accuracy(output, target, topk=(1, 2)))


def eg_5():
    """
    def train(train_loader, model, loss_fn, optimizer, epoch, args)
    """
    import time
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
        model.train()  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=train#torch.nn.Module.train
        train_loss_record = AverageMeter(name="train_loss_record")

        with tqdm(train_loader, desc="TRAIN EPOCH: {}".format(epoch)) as train_bar:
            for (data, target) in train_bar:
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                train_loss_record.update(loss.item(), data.size(0))

        writer.add_scalar("train_loss", train_loss_record.avg, epoch)


def eg_6():
    """
    def validate(val_loader, model, loss_fn, epoch, args)
    """
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

    def validate(val_loader, model, loss_fn, epoch, writer, save_dict):
        model.eval()  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval

        val_loss_record = AverageMeter(name="val_loss_record")
        with torch.no_grad():  # https://pytorch.org/docs/stable/generated/torch.no_grad.html?highlight=torch%20no_grad#torch.no_grad
            with tqdm(val_loader, desc="VALID EPOCH: {}".format(epoch)) as val_bar:
                for (data, target) in val_bar:
                    output = model(data)
                    loss = loss_fn(output, target)

                    val_loss_record.update(loss.item(), data.size(0))

        writer.add_scalar("val_loss", val_loss_record.avg, epoch)

        if val_loss_record.avg < save_dict["min"]:
            save_dict.update({"min":val_loss_record.avg, "epoch":epoch, "state_dict":model.state_dict()})


if __name__ == "__main__":
    """
    1. import argparse
    2. from torch.utils.tensorboard import SummaryWriter
       writer.add_scalar()
       writer.add_image()
       writer.add_images()
       writer.add_graph()
       tensorboard --logdir=log
    3. class AverageMeter(object)
       __dict__, __str__
    4. def accuracy(output, target, topk=(1, ))
       https://pytorch.org/docs/stable/generated/torch.topk.html#torch.topk
    5. def train(train_loader, model, loss_fn, optimizer, args)
    6. def validate(val_loader, model, loss_fn, args)
    """


    # eg_1()
    # eg_2_1()
    # eg_2_2()
    # eg_2_3()
    # eg_2_4()
    # eg_3()
    # eg_4()
    eg_5()
    # eg_6()


    print("~~~~~~下课~~~~~~")
