# Let's Go Deeper!

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        outinception_3x3_reduced,
        outinception_3x3,
        outinception_5x5_reduced,
        outinception_5x5,
        out_pool
    ):
        super().__init__()

        self.branch1 = ConvBlock(
            in_channels, out_1x1, kernel_size=1, stride=1
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, outinception_3x3_reduced, kernel_size=1),
            ConvBlock(outinception_3x3_reduced, outinception_3x3, kernel_size=3, padding=1),
        )

        # Is in the original googLeNet paper 5x5 conv but in Inception_v2 it has shown to be
        # more efficient if you instead do two 3x3 convs which is what I am doing here!
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, outinception_5x5_reduced, kernel_size=1),
            ConvBlock(outinception_5x5_reduced, outinception_5x5, kernel_size=3, padding=1),
            ConvBlock(outinception_5x5, outinception_5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        return torch.cat([y1, y2, y3, y4], 1)


class Inception(nn.Module):
    def __init__(self, img_channel):
        super().__init__()

        self.first_layers = nn.Sequential(
            ConvBlock(img_channel, 192, kernel_size=3, padding=1)
        )

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        out = self.first_layers(x)

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.max_pool(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.max_pool(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return self.fc(out)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Inception(1).to(device)
    x = torch.randn(3, 1, 32, 32, device=device)

    y: torch.Tensor = net(x)
    print(f'{y.size() = }')


if __name__ == '__main__':
    main()
