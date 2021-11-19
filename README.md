# 手把手教你跑通第一个神经网络

参考 [PYTORCH DOCUMENTATION](https://pytorch.org/docs/stable/index.html)

[toc]

## `# 1_dataset.py`

0. 继承 `torch.utils.data.Dataset`
1. 实现 `__getitem__` 和 `__len__` 两个magic methods 【个人倾向于返回字典形式】
2. 理解 `MNIST` 类，以及 `transforms` 模块
3. 利用 `torchvision.datasets` 中的数据集
4. 理解 `ImageFolder` 类及其 `classes` 与 `class_to_idx` 属性

## `# 2_dataloader.py`

0. 利用 `torch.utils.data.Dataloader`类
1. 理解 `__iter__` 这个magic method
2. 区分 `Dataloader` 与 `Dataset` 的 `__len__`
3. 利用 内置函数 `enumerate` 与 `tqdm` 模块
4. 有需要可以更改 `collate_fn`

## `# 3_model.py`

0. 继承 `torch.nn.Module`，注意 `super().__init__()`
1. 理解 `__call__` 这个magic method 与自定义 `forward` 关系
2. 注意 `PyTorch` 中数据的摆放 `(B, C, H ,W)`
3. 调用 `torchvison.models` 中现成的网络
4. 注意 `torch.nn.Module.dict_state()` `torch.save()` `torch.load` 以及 `torch.nn.Module.load_state_dict()` 及其中参数
5. 利用 `torch.utils.model_zoo.load_url()` 下载预训练参数

## `# 4_optimizer.py`

0. 调用 `torch.optim` 模块中的优化器
1. 注意 `params`参数
2. 通过 `optimizer.zero_grad()` `loss.backward()` `optimizer.step()` 开始训练

## `# 5_train.py`

- [x] 综上所述，完成训练！

- [ ] 美化代码，下次一定！
