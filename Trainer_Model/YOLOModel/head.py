import torch
import torch.nn as nn
from YOLOModel import Conv, DFL, Params
class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch*4        #number of bounding box coordinates
        self.nc = num_classes               #80 for coco change to number of classes in our data set
        self.no = self.coordinates+self.nc  #number of outputs per anchor box
        self.stride = torch.zeros(3)        #strides computed during build
        yolo_params = Params(version=version)
        d,w,r = yolo_params.return_params()

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