import os
from PIL import Image
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
torch.manual_seed(0)


class FlameDataset(Dataset):
    """
    火焰目标检测
    """
    def __init__(self, dataset_path, transform=None, mode="train"):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, "JPEGImages")
        self.target_path = os.path.join(dataset_path, "Annotations")
        assert mode in ["test", "train", "trainval", "val"]
        self.index_list_path = os.path.join(dataset_path, "ImageSets", "Main", mode+".txt")
        with open(self.index_list_path, "r") as f:
            self.index_list = [l.strip() for l in f.readlines()]
        self.length = len(self.index_list)
        self.transform = transform

    def simple_parse_xml(self, target_name):
        tree = ET_parse(target_name)
        labels = []
        boxes = []
        for elem in tree.iter():
            if elem.tag == "object":
                label = int(1)  # 只检测火焰
                labels.append(label)
                xmin, ymin, xmax, ymax = None, None, None, None
                for e in list(elem):
                    if e.tag == "bndbox":
                        for ee in list(e):
                            if ee.tag == "xmin":
                                xmin = int(ee.text)
                            elif ee.tag == "xmax":
                                xmax = int(ee.text)
                            elif ee.tag == "ymin":
                                ymin = int(ee.text)
                            elif ee.tag == "ymax":
                                ymax = int(ee.text)
                boxes.append([xmin, ymin, xmax, ymax])

        # return {"labels":labels, "boxes":torch.Tensor(boxes)}
        return {"labels":torch.as_tensor(labels, dtype=torch.int64),
                "boxes":torch.as_tensor(boxes, dtype=torch.float)}

    def __getitem__(self, index):
        image_name = os.path.join(self.image_path, self.index_list[index]+".jpg")
        target_name = os.path.join(self.target_path, self.index_list[index]+".xml")

        image = Image.open(image_name).convert("RGB")
        target = self.simple_parse_xml(target_name)

        if self.transform:
            image = self.transform(image)
        return {"image":image, "target":target}

    def __len__(self):
        return self.length

    def collate_fn_(self, batch):
        """
        before_collate:
            [
                {"image":Tensor1, "target":{"labels":labels1, "boxes":boxes1}},
                ...
                {"image":TensorN, "target":{"labels":labelsN, "boxes":boxesN}},
            ]
        after_collate:
            {
                "images":[Tensor1, ..., TensorN]  # 每一个Tensor都是[C, H, W], 取值0-1, 但是高和宽可以不同
                "targets":{
                    "target":[labels1, ..., labelsN],
                    "boxes":[boxes1, ..., boxesN]
                    }
            }
        """
        images = [i["image"] for i in batch]
        targets = [{"labels":i["target"]["labels"], "boxes":i["target"]["boxes"]} for i in batch]

        return {"images":images, "targets":targets}


def eg_1():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (255, 255, 255))
        ]
    )
    dataset_path = "./flame_dataset"
    flame_dataset = FlameDataset(dataset_path=dataset_path, transform=transform, mode="trainval")
    flame_dataloader = DataLoader(flame_dataset, batch_size=4, shuffle=True, collate_fn=flame_dataset.collate_fn_)
    model = fasterrcnn_resnet50_fpn(num_classes=1+1, pretrained=False, pretrained_backbone=False)

    model.train()
    for batch in flame_dataloader:
        pred = model(batch["images"], batch["targets"])
        print(pred)
        break


if __name__ == "__main__":
    """
    1. 创建 CancerDataset 类
        __init__ : 存路径，不存图片
        __getitem__ : 按索引返回值
        __len__ : 返回样本数量
    2. 跑一个简单的网络
        测试 dataset, dataloader, model
        可以参考 https://www.bilibili.com/video/BV1UR4y1t7Cm/
        数据集分割
        可以参考 https://www.bilibili.com/video/BV1fS4y1Q7Ph/
    """
    eg_1()

    print("~~~~~~下课~~~~~~")