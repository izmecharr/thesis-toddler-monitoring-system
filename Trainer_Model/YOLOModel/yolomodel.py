from torch.utils import data
import torch
import torch.nn as nn
import os
import yaml
    
class Conv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1,padding=1,groups=1,activation=True):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False,groups=groups)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001,momentum=0.03)
        self.act=nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

#2.1 Bottleneck: stack of 2 Conb with shortcut connection ( True/ False )

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
    
#2.2 C2f: Convolution + bottleneck*N + Conv
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


#SPPF
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
    


#bakcbone = DarkNet53

#return depth(d), width(w), ratio(r) based on version
def yolo_params(version):
    if version == 'n':
        return 1/3,1/4,2.0
    elif version == 's':
        return 1/3,1/2,2.0
    elif version == 'm':
        return 2/3,3/2,1.5
    elif version == 'l':
        return 1.0,1.0,1.0
    elif version == 'x':
        return 1.0,1.25,1.0
    
class Backbone(nn.Module):
    def __init__(self,version,in_channels=3,shortcut=True):
        super().__init__()
        d,w,r = yolo_params(version)

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
print("\n-----------------------------------------")
print("--------------sanity check--------------")
print("---------backbone forward pass----------")
print("-----------------------------------------\n")

print("----Nano model----")
backbone_n = Backbone(version='n')
print(f"{sum(p.numel() for p in backbone_n.parameters())/1e6} million parameters")  
print("----Small model----")
backbone_s = Backbone(version='s')
print(f"{sum(p.numel() for p in backbone_s.parameters())/1e6} million parameters")  

x = torch.rand((1,3,640,640))
out1,out2,out3 = backbone_n(x)

print(out1.shape)
print(out2.shape)
print(out3.shape)

#Upsample

class Upsample(nn.Module):
    def __init__(self,scale_factor=2,mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self,x):
        return nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)

#NECK
class Neck(nn.Module):
    def __init__(self,version):
        super().__init__()
        d,w,r = yolo_params(version)

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
    

#sanity check
print("\n-----------------------------------------")
print("--------------sanity check--------------")
print("------------neck forward pass-----------")
print("-----------------------------------------\n")

neck = Neck(version='n')
print(f"{sum(p.numel() for p in neck.parameters())/1e6} million parameters")

x = torch.rand((1,3,640,640))

out1,out2,out3 = Backbone(version='n')(x)

out_1,out_2,out_3 = neck(out1,out2,out3)
print(out_1.shape)
print(out_2.shape)
print(out_3.shape)
        
#head

class DFL(nn.Module):
    def __init__(self,ch=16):
        super().__init__()

        self.ch=ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)

        #initialize conv with [0, ..., ch-1]
        x = torch.arange(ch, dtype=torch.float).view(1,ch,1,1)

        self.conv.weight.data[:]=torch.nn.Parameter(x) #DFL only has ch parameters

    def forward(self, x):
        #x must have num_channels = 4*ch: x = [bs, 4*ch, c]
        b,c,a = x.shape     #c = 4*ch

        x = x.view(b,4,self.ch,a).transpose(1,2) #[bs,ch,4,a]

        #take softmax on channel dimension to get distribution probablities
        x = x.softmax(1)        # [b, ch, 4, a]
        x = self.conv(x)        # [b, 1, 4, a]
        return x.view(b,4,a)    # [b, 4, a]
    
#sanity check
print("\n-----------------------------------------")
print("--------------sanity check--------------")
print("-------------DFL forward pass-----------")
print("-----------------------------------------\n")
dummy_input = torch.rand((1,64,128))
dfl = DFL() 
    
print(f"{sum(p.numel()for p in dfl.parameters())} parameters")

dummy_output = dfl(dummy_input)

print(dummy_output.shape)


class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch*4        #number of bounding box coordinates
        self.nc = num_classes               #80 for coco change to number of classes in our data set
        self.no = self.coordinates+self.nc  #number of outputs per anchor box
        self.stride = torch.zeros(3)        #strides computed during build
        d,w,r = yolo_params(version=version)

        self.box = nn.ModuleList([
            nn.Sequential(Conv(int(256*w),self.coordinates,kernel_size=3,stride=1,padding=1), Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1), nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w),self.coordinates,kernel_size=3,stride=1,padding=1),Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1), nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)),

            nn.Sequential(Conv(int(512*w*r), self.coordinates,kernel_size=3,stride=1,padding=1), Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1), nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1))
        ])

        self.cls=nn.ModuleList([
            nn.Sequential(Conv(int(256*w), self.nc, kernel_size=3,stride=1,padding=1),
                          Conv(self.nc, self.nc,kernel_size=3,stride=1,padding=1),nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w), self.nc, kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)),
            nn.Sequential(Conv(int(512*w*r),self.nc,kernel_size=3,stride=1,padding=1),
                          Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                          nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1))
        ])

        self.dfl=DFL()

    def forward(self,x):

        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            x[i] = torch.cat((box,cls), dim=1)

        if self.training:
            return x
        
        anchors, strides = (i.transpose(0,1) for i in self.make_anchors(x, self.stride))

        x = torch.cat([i.view(x[0].shape[0], self.no,-1) for i in x], dim = 2)
        
        box, cls = x.split(split_size=(4*self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2,1)

        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a+b) / 2, b - a),dim=1)
        
        return torch.cat(tensors=(box * strides, cls.sigmoid()),dim=1)

    def make_anchors(self, x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape

            sx = torch.arange(end=w, device= device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device,dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx)

            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1,2))

            stride_tensor.append(torch.full((h*w,1),stride,dtype=dtype,device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)

#sanity check
print("\n-----------------------------------------")
print("--------------sanity check--------------")
print("------------------Head------------------")
print("-----------------------------------------\n")
detect = Head(version='n')
print(f"{sum(p.numel() for p in detect.parameters())/1e6} million parameters")

output = detect([out_1,out_2,out_3])

print(f"output1: {output[0].shape}")
print(f"output2: {output[1].shape}")
print(f"output3: {output[2].shape}")

class MyYolo(nn.Module):
    def __init__(self,version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version = version)
        self.head = Head(version=version)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0],x[1],x[2])
        return self.head(list(x))


#sanity check
print("\n-----------------------------------------")
print("--------------sanity check--------------")
print("---------------full model---------------")
print("-----------------------------------------\n")
model = MyYolo(version='n')
print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")