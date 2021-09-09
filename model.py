import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 9), # 64@120*120
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),# 64 @ 60 * 60
            nn.Conv2d(64, 128, 7), # 128 @ 54 * 54
            nn.ReLU(),    
            nn.MaxPool2d(2), # 128 @ 27 * 27
            nn.Conv2d(128, 128, 4), # 128 @ 24 * 24
            nn.ReLU(), 
            nn.MaxPool2d(2),# 128 @ 12 * 12
            nn.Conv2d(128, 256, 4), # 256 @ 9 * 9 
            nn.ReLU(), 
        )
        self.liner = nn.Sequential(nn.Linear(20736, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

# for test
if __name__ == '__main__':
    net = Siamese()
    print(net)
    print(list(net.parameters()))
