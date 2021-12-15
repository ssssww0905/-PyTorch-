import torch
import torch.nn as nn
torch.manual_seed(1)


def eg_1():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 3, 3]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]), bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    for name, param in layer.named_parameters():
        print(name, param)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_2():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 1, 7, 7] -> out.shape [1, 1, 5, 5]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1))

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[1., 0., 2., 1., 1., 0., 1.],
                        [0., 1., 1., 1., 2., 2., 3.],
                        [3., 2., 2., 1., 3., 2., 1.],
                        [2., 3., 4., 5., 1., 1., 0.],
                        [1., 1., 2., 3., 4., 1., 2.],
                        [2., 3., 2., 1., 3., 1., 1.],
                        [2., 1., 2., 1., 2., 1., 1.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_3():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (2, 2)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 2, 2]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(2, 2))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    for name, param in layer.named_parameters():
        print(name, param)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_4():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (3, 3)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 1, 1]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(3, 3))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    for name, param in layer.named_parameters():
        print(name, param)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_5():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (3, 3)
    padding = (1, 1)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 2, 2]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(3, 3),
                      padding=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    for name, param in layer.named_parameters():
        print(name, param)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_6():
    """
    in_channels = 1
    out_channels = 3
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 1, 5, 5] -> out.shape [1, 3, 3, 3]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=3,
                      kernel_size=(3, 3),
                      stride=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([3, 1, 3, 3]), bias torch.Size([3])


    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]],
                           [[[1., 1., 0],
                           [0, 0., 1.],
                           [1., 1., 1.]]],
                           [[[0., 2., 0],
                           [0, 0., 1.],
                           [1., 0., 0.]]],
                           ])
    bias = torch.Tensor([5., 1., 0.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_7():
    """
    in_channels = 2
    out_channels = 3
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 2, 5, 5] -> out.shape [1, 3, 3, 3]
    """
    layer = nn.Conv2d(in_channels=2,
                      out_channels=3,
                      kernel_size=(3, 3),
                      stride=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([3, 2, 3, 3]), bias torch.Size([3])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]],
                            [[1., 1., 0],
                           [0, 1., 1.],
                           [0., 0., 2.]]],
                           [[[1., 1., 0],
                           [0, 0., 1.],
                           [1., 1., 1.]],
                            [[1., 1., 0],
                           [0, 1., 1.],
                           [1., 0., 1.]]],
                           [[[0., 2., 0],
                           [0, 0., 1.],
                           [1., 0., 0.]],
                            [[2., 2., 0],
                           [0, 2., 2.],
                           [2., 0., 2.]]]])
    bias = torch.Tensor([5., 1., 0.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]],
                      [[0., 3., 4., 3., 2.],
                       [1., 0., 3., 4., 1.],
                       [1., 4., 0., 2., 1.],
                       [4., 7., 5., 0., 2.],
                       [1., 1., 4., 3., 1.]]]])
    print(x.shape)
    out = layer(x)
    print("out:\n{}".format(out))


def ex():
    print("~~~~~~作业~~~~~~")
    print("layer = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)")
    print("x = torch.randn((128, 3, 224, 224))")
    print("out = layer(x)")

    print("(1)layer.weight.shape?")
    print("(2)layer.bias.shape?")
    print("(3)out.shape?")


if __name__ == "__main__":
    """
    nn.Conv2d 继承于 nn.Module,
    所以可以像调用函数一样调用类的实例

    顾名思义，Conv2d 就是二维的卷积
    输入的 tensor 必须是 (N, C_{in}, H, W)

    输出的 tensor ？？？
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
    """

    # eg_1()
    # eg_2()
    # eg_3()
    # eg_4()
    # eg_5()
    # eg_6()
    # eg_7()

    ex()
    print("~~~~~~下课~~~~~~")