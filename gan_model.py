import torch
from torch import nn
from torchsummary import summary
import copy

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dense1 = nn.Sequential(
                nn.Conv2d(in_channels=172, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense2 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense5 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense6 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(6,6)),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)

        return x


class Generative(nn.Module):
    def __init__(self):
        super(Generative, self).__init__()

        self.dense1 = nn.Sequential(
                nn.Conv2d(in_channels=172, out_channels=172, kernel_size=(7,7)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense2 = nn.Sequential(
                nn.Conv2d(in_channels=172, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.block1 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=1, padding=1),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=2, padding=2)
                )

        self.block2 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=3, padding=3),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=4, padding=4)
                )

        self.block3 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=5, padding=5),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=6, padding=6)
                )

        self.block4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=7, padding=7),
                nn.LeakyReLU(negative_slope=0.05, inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), dilation=8, padding=8)
                )

        self.dense4 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense5 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4,4)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.dense6 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=172, kernel_size=(7,7)),
                nn.LeakyReLU(negative_slope=0.05, inplace=True)
                )

        self.leakyrelu = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.05)
                )


    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        b1 = self.block1(x)
        x1 = self.leakyrelu(b1 + x)
        b2 = self.block2(x1)
        x2 = self.leakyrelu(b2 + x1)
        b3 = self.block3(x2)
        x3 = self.leakyrelu(b3 + x2)
        b4 = self.block4(x3)
        x = self.leakyrelu(b4 + x3)
        
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)

        return x

if __name__=="__main__":
    model = Discriminator()
    summary(model, (172, 256, 256))

    gmodel = Generative()
    summary(gmodel, (172, 256, 256))
