import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv, DFL, Detect

class GroupNormConv(nn.Module):
    """Conv module with GroupNorm instead of BatchNorm for small feature maps."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        # Use GroupNorm instead of BatchNorm
        num_groups = min(max(2, c2 // 4), 32)
        self.norm = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualC2f(nn.Module):
    """C2f block with residual connection for better gradient flow."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c2f = C2f(c1, c2, n, shortcut, g, e)  # Use standard C2f
        self.residual = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        
    def forward(self, x):
        return self.c2f(x) + self.residual(x)


class SmallObjectEnhance(nn.Module):
    """Channel attention module optimized for small object detection with GroupNorm."""
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.cv1 = GroupNormConv(c1, c2//2, 1, 1, act=act)
        self.cv2 = GroupNormConv(c2//2, c2, 3, 1, act=act)
        
        # Attention mechanism
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            GroupNormConv(c2, c2//16, 1, act=act),
            GroupNormConv(c2//16, c2, 1, act=act),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.cv2(self.cv1(x))
        return x * self.attn(x)