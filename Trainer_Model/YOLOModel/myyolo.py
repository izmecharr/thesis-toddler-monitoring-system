import torch.nn as nn
from YOLOModel import Backbone, Neck, Head
class MyYolo(nn.Module):
    def __init__(self,version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0],x[1],x[2])
        return self.head(list(x))
    

# if __name__ == '__main__':
#     #sanity check
#     print("\n-----------------------------------------")
#     print("--------------sanity check--------------")
#     print("---------------full model---------------")
#     print("-----------------------------------------\n")
#     model = MyYolo(version='n')
#     print(f"{sum(p.numel() for p in model.parameters())/1e6} million parameters")
