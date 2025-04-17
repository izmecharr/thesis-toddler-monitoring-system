import torch.nn as nn
import torch
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