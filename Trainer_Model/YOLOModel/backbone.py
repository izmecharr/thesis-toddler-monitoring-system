import torch.nn as nn
from YOLOModel import Conv, C2f, SPPF, Params
    
class Backbone(nn.Module):
    def __init__(self,version,in_channels=3,shortcut=True):
        super().__init__()
        yolo_params = Params(version)
        d,w,r = yolo_params.return_params()

        #conv layers
        self.conv_0 = Conv(in_channels,int(64*w),kernel_size=3,stride=2,padding=1)
        self.conv_1 = Conv(int(64*w),int(128*w),kernel_size=3,stride=2,padding=1)
        self.conv_3 = Conv(int(128*w),int(256*w),kernel_size=3,stride=2,padding=1)
        self.conv_5 = Conv(int(256*w),int(512*w),kernel_size=3,stride=2,padding=1)
        self.conv_7 = Conv(int(512*w),int(512*w*r),kernel_size=3,stride=2,padding=1)

        #c2f layers
        self.c2f_2 = C2f(int(128*w),int(128*w),num_bottlenecks=int(3*d),shortcut=True)
        self.c2f_4 = C2f(int(256*w),int(256*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_6 = C2f(int(512*w),int(512*w),num_bottlenecks=int(6*d),shortcut=True)
        self.c2f_8 = C2f(int(512*w*r),int(512*w*r),num_bottlenecks=int(3*d),shortcut=True)

        #sppf
        self.sppf = SPPF(int(512*w*r),int(512*w*r))

    def forward(self,x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x) # keep for output
        x = self.conv_5(out1)
        out2 = self.c2f_6(x) # keep for output

        print(f"\n{out2.shape}\n")

        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)
        return out1,out2,out3
    
#sanity check
# print("\n-----------------------------------------")
# print("--------------sanity check--------------")
# print("---------backbone forward pass----------")
# print("-----------------------------------------\n")

# print("----Nano model----")
# backbone_n = Backbone(version='n')
# print(f"{sum(p.numel() for p in backbone_n.parameters())/1e6} million parameters")  
# print("----Small model----")
# backbone_s = Backbone(version='s')
# print(f"{sum(p.numel() for p in backbone_s.parameters())/1e6} million parameters")  

# x = torch.rand((1,3,640,640))
# out1,out2,out3 = backbone_n(x)

# print(out1.shape)
# print(out2.shape)
# print(out3.shape)