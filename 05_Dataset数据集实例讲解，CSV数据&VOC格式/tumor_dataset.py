import os
import pandas as pd
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)

def eg_0_0():
    """
    read feature.csv
    https://pandas.pydata.org/
    """
    feature_path = "./tumor_dataset/expression.csv"
    feature_df = pd.read_csv(feature_path)  # 89个样本
    print("type(feature_df): {}".format(type(feature_df)))  # <class 'pandas.core.frame.DataFrame'>
    print("feature_df.shape: {}".format(feature_df.shape))  # (56602, 90)
    print(feature_df.columns.tolist())
    print("========================== INFO ==========================")
    feature_df.info()


def eg_0_1():
    """
    read label.csv
    """
    label_path = "./tumor_dataset/annotation.csv"
    label_df = pd.read_csv(label_path)  # 89个样本
    print("type(label_df): {}".format(type(label_df)))  # <class 'pandas.core.frame.DataFrame'>
    print("label_df.shape: {}".format(label_df.shape))  # (89, 2)
    print(label_df[label_df.columns[0]].tolist())
    print("========================== INFO ==========================")
    label_df.info()


class TumorDataset(Dataset):
    """
    肿瘤分类
    """
    def __init__(self, dataset_path):
        # super().__init__()  # 疑问
        self.dataset_path = dataset_path
        self.feature_path = os.path.join(self.dataset_path, "expression.csv")
        self.label_path = os.path.join(self.dataset_path, "annotation.csv")
        assert os.path.exists(self.feature_path), "expression.csv not exist"
        assert os.path.exists(self.label_path), "annotation.csv not exist"
        feature_df_ = pd.read_csv(self.feature_path)
        label_df_ = pd.read_csv(self.label_path)
        assert feature_df_.columns.tolist()[1:] == label_df_[label_df_.columns[0]].tolist(),\
            "feature name does not match label name"
        self.feature = [feature_df_[i].tolist() for i in feature_df_.columns[1:]]
        self.label = [1 if label_df_.iat[i, 1] == "Tumor" else 0 for i in label_df_.index]
        assert len(self.feature) == len(self.label)
        self.length = len(self.feature)

    def __getitem__(self, index):
        x = self.feature[index]  # <class 'list'>, 56602
        x = torch.Tensor(x)  # <class 'torch.Tensor'>, torch.Size([56602])

        y = self.label[index]  # <class 'int'>
        return {"x":x, "y":y}  # DataLoader 默认的 collate_fn 会将 int -> Tensor

    def __len__(self):
        return self.length


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


def eg_1():
    """
    train & test
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()  # 必要的
            self.layers = nn.Sequential(
                nn.Linear(56602, 100),  # 特征向量56602维
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(100, 2)  # 2分类问题
            )
        def forward(self, x):
            return self.layers(x)

    dataset_path = "./tumor_dataset"
    # dataset
    tumor_dataset = TumorDataset(dataset_path=dataset_path)
    # dataloader
    tumor_loader = DataLoader(tumor_dataset, batch_size=4, shuffle=True, collate_fn=None)
    # model
    model = SimpleModel().to(device)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()
    # train
    for epoch in range(5):
        training_loss = AverageMeter("training_loss")  # 之前视频有解释过
        training_acc1 = AverageMeter("training_acc1")  # 之前视频有解释过
        for batch in tumor_loader:
            model.train()  # 之前视频有解释过
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            acc1 = accuracy(pred, y, topk=(1,))
            training_acc1.update(acc1[0].item(), len(x))  # 之前视频有解释过

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.update(loss.item(), len(x))  # 之前视频有解释过
        print("epoch: {} | training_loss: {:.6f}, training_acc1: {:.6f}"\
            .format(epoch, training_loss.avg, training_acc1.avg))
    # test
    test_dataset = TumorDataset(dataset_path=dataset_path)
    test_loader = DataLoader(tumor_dataset, batch_size=len(test_dataset), shuffle=False)
    for batch in test_loader:  # 只有一个batch
        model.eval()  # 之前视频有解释过
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred_tensor = model(x)
        pred_label = pred_tensor.max(dim=1)[1].cpu().numpy()
        pred_label = ["tumor" if i == 1 else "normal" for i in pred_label]
    np.savetxt(os.path.join(dataset_path, "pred_label.csv"), pred_label, fmt="%s", delimiter=",")


if __name__ == "__main__":
    """
    0. 利用pandas库读取 csv 数据
    1. 创建 TumorDataset 类
        __init__ : 读取csv
        __getitem__ : 按索引返回值
        __len__ : 返回样本数量
    2. 跑一个简单的网络
        dataset, dataloader, model, optim, train
        可以参考 https://www.bilibili.com/video/BV1UR4y1t7Cm/
        AverageMeter, accuracy, model.train(), model.eval()
        可以参考 https://www.bilibili.com/video/BV1fS4y1Q7Ph/
    """
    eg_0_0()
    eg_0_1()
    eg_1()

    print("~~~~~~下课~~~~~~")
