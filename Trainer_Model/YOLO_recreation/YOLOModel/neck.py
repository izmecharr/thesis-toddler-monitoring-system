import torch.nn as nn
import torch
from . import Conv, C2f, Upsample, Params
#NECK
class Neck(nn.Module):
    def __init__(self,version):
        super().__init__()
        yolo_params = Params(version)
        d,w,r = yolo_params.return_params()

        self.up = Upsample() #no trainable parameters
        self.c2f_1 = C2f(in_channels=int(512*w*(1+r)),out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_2 = C2f(in_channels=int(768*w),out_channels=int(256*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_3 = C2f(in_channels=int(768*w),out_channels=int(512*w),num_bottlenecks=int(3*d),shortcut=False)
        self.c2f_4 = C2f(in_channels=int(512*w*(1+r)),out_channels=int(512*w*r),num_bottlenecks=int(3*d),shortcut=False)

        self.conv_1 = Conv(in_channels=int(256*w), out_channels=int(256*w), kernel_size=3,stride=2,padding=1)
        self.conv_2 = Conv(in_channels=int(512*w), out_channels=int(512*w), kernel_size=3,stride=2,padding=1)

    def forward(self, x_res_1, x_res_2, x): #res_1 c2f_4 ; res_2 c2f_6 ; res_3 sppf
        res_1 = x

        x = self.up(x)
        x = torch.cat([x,x_res_2], dim=1)
        res_2 = self.c2f_1(x)
        x = self.up(res_2)
        x = torch.cat([x,x_res_1], dim=1)
        out_1 = self.c2f_2(x)
        x = self.conv_1(out_1)
        x = torch.cat([x,res_2], dim=1)
        out_2 = self.c2f_3(x)
        x = self.conv_2(out_2)
        x = torch.cat([x,res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3