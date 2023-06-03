import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized (But not similar to vision adapter)
    """

    def __init__(self, input_size, reduction_factor=16):
        super().__init__()
        self.input_size = input_size
        self.reduction_factor = reduction_factor

        self.down_sample_size = self.input_size // self.reduction_factor
        self.activation = nn.ReLU(inplace=True)
        self.down_sampler = nn.Linear(self.input_size, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_size)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)

        return output

class VisionAdapter(nn.Module):
    """
    Vision Adapter layer, the down_sampler and up_sampler modules are Convolution instead of Linear.
    Normally, the weights of these modules are not optimized.
    And, the down_sampler module is a 1x1 Convolution with stride 2.
    """
    def __init__(self, input_dim, output_dim, adapter_kind, reduction_factor=16, use_bn=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn

        if self.adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

        elif self.adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm2d(output_dim)

        else:
            raise NotImplementedError("Adapter kind {} is not implemented".format(self.adapter_kind))

    def forward(self, x):
        if self.adapter_kind == "bottleneck":
            z = self.down_sampler(x)
            z = self.bn1(z) if self.use_bn else z
            z = self.activation(z)
            output = self.up_sampler(z)
            output = self.bn2(output) if self.use_bn else output

        elif self.adapter_kind == "basic":
            z = self.conv(x)
            z = self.bn(z) if self.use_bn else z

        return output