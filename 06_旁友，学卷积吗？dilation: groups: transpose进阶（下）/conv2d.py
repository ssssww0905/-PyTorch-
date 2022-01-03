import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d


def eg_1():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    dilation = (1, 1)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 3, 3]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      dilation=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

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
    dilation = (2, 2)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 1, 1]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      dilation=(2, 2))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_3():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    dilation = (2, 1)

    x.shape [1, 1, 5, 5] -> out.shape [1, 1, 1, 3]
    """
    layer = nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      dilation=(2, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 3., 4., 3., 2.],
                       [1., 2., 3., 4., 5.],
                       [6., 4., 3., 2., 1.],
                       [4., 7., 5., 8., 1.],
                       [6., 5., 4., 3., 2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_4():
    """
    in_channels = 6
    out_channels = 9
    kernel_size = (4, 4)
    stride = (1, 1)
    groups = 1

    x.shape [1, 6, 5, 5] -> out.shape [1, 9, 2, 2]
    """
    layer = nn.Conv2d(in_channels=6,
                      out_channels=9,
                      kernel_size=(4, 4),
                      stride=(1, 1),
                      groups=1)
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([9, 6, 4, 4]), bias torch.Size([9])

    x = torch.randn((1, 6, 5, 5))
    out = layer(x)
    print("out.shape: {}".format(out.shape))  # out.shape: torch.Size([1, 9, 2, 2])


def eg_5():
    """
    in_channels = 6
    out_channels = 9
    kernel_size = (4, 4)
    stride = (1, 1)
    groups = 3

    x.shape [1, 6, 5, 5] -> out.shape [1, 9, 2, 2]
    """
    layer = nn.Conv2d(in_channels=6,
                      out_channels=9,
                      kernel_size=(4, 4),
                      stride=(1, 1),
                      groups=3)
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([9, 2, 4, 4]), bias torch.Size([9])

    x = torch.randn((1, 6, 5, 5))
    out = layer(x)
    print("out.shape: {}".format(out.shape))  # out.shape: torch.Size([1, 9, 2, 2])


def eg_6():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 1, 1, 1] -> out.shape [1, 1, 3, 3]
    """
    layer = nn.ConvTranspose2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_7():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)

    x.shape [1, 1, 2, 2] -> out.shape [1, 1, 4, 4]
    """
    layer = nn.ConvTranspose2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 1.],
                        [1., 0.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_8():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (3, 3)

    x.shape [1, 1, 2, 2] -> out.shape [1, 1, 6, 6]
    """
    layer = nn.ConvTranspose2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(3, 3))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2., 1.],
                        [1., 0.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


def eg_9():
    """
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    dilation = (2, 2)

    x.shape [1, 1, 1, 1] -> out.shape [1, 1, 5, 5]
    """
    layer = nn.ConvTranspose2d(in_channels=1,
                      out_channels=1,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      dilation=(2, 2))
    for name, param in layer.named_parameters():
        print(name, param.data.shape)  # weight torch.Size([1, 1, 3, 3]) bias torch.Size([1])

    weight = torch.Tensor([[[[1., 2., 0],
                           [0, 2., 1.],
                           [2., 1., 2.]]]])
    bias = torch.Tensor([5.])
    state_dict = {"weight":weight, "bias":bias}
    layer.load_state_dict(state_dict=state_dict)

    x = torch.Tensor([[[[2.]]]])
    out = layer(x)
    print("out:\n{}".format(out))


if __name__ == "__main__":
    """
    1.Conv2d
        Input: (N, C_{in}, H_{in}, W_{in})
        Output: ???
        Args: in_channels, out_channels, kernel_size, stride, padding
        Args: dilation, groups

    2. ConvTranspose2d
        Input: (N, C_{in}, H_{in}, W_{in})
        Output: ???
        Args: in_channels, out_channels, kernel_size, stride, padding, dilation, groups

    gif:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    # eg_1()
    # eg_2()
    # eg_3()
    # eg_4()
    # eg_5()
    # eg_6()
    # eg_7()
    # eg_8()
    # eg_9()

    print("~~~~~~下课~~~~~~")