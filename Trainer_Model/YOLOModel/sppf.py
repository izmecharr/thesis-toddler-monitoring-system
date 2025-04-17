import torch.nn as nn
import torch
from . import Conv

class SPPF(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=5):
        #kernel_size = size of maxpool
        super().__init__()
        hidden_channels = in_channels//2
        self.conv1 = Conv(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        
        # concatenate outputs of maxpool and feed to conv2
        self.conv2 = Conv(4*hidden_channels,out_channels,kernel_size=1,stride=1,padding=0)

        #maxpool is applied at 3 different scales
        self.m = nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=kernel_size//2,dilation=1,ceil_mode=False)

    def forward(self,x):
        x = self.conv1(x)

        #apply maxpooling at different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        #concatenate
        y = torch.cat([x,y1,y2,y3], dim=1)

        #final conv
        y = self.conv2(y)

        return y
    
#sanity check AFTER C2F DECLARATION
# print("\n-----------------------------------------")
# print("--------------sanity check--------------")
# print("-----------------sppf-------------------")
# print("-----------------------------------------\n")

# sppf = SPPF(in_channels=128,out_channels=512)
# print(f"{sum(p.numel() for p in sppf.parameters())/1e6} million parameters")

# dummy_input = sppf(dummy_input)

# print("Output shape: ", dummy_input.shape)