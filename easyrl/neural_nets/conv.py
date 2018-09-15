
import torch
import torch.nn as nn


# wrapper for nn.Conv2D that calculates 'same' padding automatically
def conv(in_shape, c_in, c_out, kernel_size=3, stride=1, padding ='same'):
    def get_tuple(z):
        if isinstance(z, tuple):
            return z
        else:
            return z, z
    kernel_size = get_tuple(kernel_size)
    stride = get_tuple(stride)
    if padding == 'same':
        # formula for 'same' padding
        padding = tuple(((stride[i] * (in_shape[i] - 1) - in_shape[i] + kernel_size[i]) // 2) for i in (0,1))

    conv_layer = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)

    out_shape = tuple((((in_shape[i] + 2 * padding[i] - kernel_size [i]) // stride[i] + 1) for i in (0,1)))

    conv_layer = nn.Sequential(conv_layer,  nn.BatchNorm2d(c_out), nn.MaxPool2d(kernel_size=2, stride=2),  nn.ReLU())
    out_shape = tuple(((out_shape[i] - kernel_size[i]) // stride[i] + 1) for i in (0,1))

    return conv_layer, out_shape