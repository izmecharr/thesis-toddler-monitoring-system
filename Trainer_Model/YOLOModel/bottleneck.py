from YOLOModel import Conv
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,shortcut=True):
        super().__init__()
        self.conv1=Conv(in_channels, out_channels,kernel_size=3,stride=1,padding=1)

        self.conv2=Conv(out_channels, out_channels,kernel_size=3,stride=1,padding=1)
        self.shortcut=shortcut

    def forward(self, x):
        x_in = x #for residual connection
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in
        return x