import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MOBILENETV2(nn.Module):
    def __init__(self):
        super(MOBILENETV2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(32, 32, 2, False, 2)

        self.block3_2 = InvertedResidual(32, 32, 1, True, 2)
        self.block3_3 = InvertedResidual(32, 32, 1, True, 2)
        self.block3_4 = InvertedResidual(32, 32, 1, True, 2)
        self.block3_5 = InvertedResidual(32, 32, 1, True, 2)

        self.conv4_1 = InvertedResidual(32, 64, 2, False, 2)

        self.conv5_1 = InvertedResidual(64, 64, 1, False, 4)
        self.block5_2 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_3 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_4 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_5 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_6 = InvertedResidual(64, 64, 1, True, 4)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        return x




class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        self.mobilenetv2 = MOBILENETV2()


        self.FC = nn.Sequential(
            nn.Linear(64*5*8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) )


        self.output = nn.Sequential(
            nn.Linear(64+2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2),)



    def forward(self, x_in):
        feature = self.mobilenetv2(x_in['eye'])
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)

        feature = torch.cat((feature, x_in['head_pose']), 1)
        gaze = self.output(feature)

        return gaze


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    m = model().to(device)
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2).to(device),
               "eye": torch.zeros(10, 3, 36, 60).to(device)
               }
    a = m(feature)
    print(m)

