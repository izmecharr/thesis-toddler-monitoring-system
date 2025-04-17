import torch.nn as nn
import torch
from . import Conv, Bottleneck

class C2f(nn.Module):
    def __init__(self,in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels//2
        self.num_bottlnecks = num_bottlenecks
        self.conv1 = Conv(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

        #sequence of bottleneck layers
        self.m=nn.ModuleList([Bottleneck(self.mid_channels,self.mid_channels) for _ in range(num_bottlenecks)])
        self.conv2 = Conv((num_bottlenecks+2)*out_channels//2,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)

        #split x along channel dimension
        x1,x2=x[:,:x.shape[1]//2,:,:],x[:,x.shape[1]//2:,:,:]
        #list of outputs
        outputs=[x1,x2] #x1 is fed through the bottlenecks

        for i in range(self.num_bottlnecks):
            x1 = self.m[i](x1) #[bs, 0.5c_out, w, h]
            outputs.insert(0,x1)
        
        outputs = torch.cat(outputs,dim=1) #[bs,0.5c_out(num_bottleneck+2),w,h]
        out = self.conv2(outputs)

        return out
    

    

#sanity check AFTER CONV AND BOTTLENECK DECLARATION
# print("\n-----------------------------------------")
# print("--------------sanity check--------------")
# print("----------conv bottleneck c2f------------")
# print("-----------------------------------------\n")

# c2f = C2f(in_channels=64,out_channels=128,num_bottlenecks=2)
# print(f"{sum(p.numel() for p in c2f.parameters())/1e6} million parameters")

# dummy_input = torch.rand((1,64,244,244))

# dummy_input = c2f(dummy_input)

# print("Output shape: ", dummy_input.shape)