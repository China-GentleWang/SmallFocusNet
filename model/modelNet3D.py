import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SEBlock
        out = self.se(out)

        out += identity
        out = self.relu2(out)
        return out


class smallFocusNet(nn.Module):
    def __init__(self, num_classes=3):
        super(smallFocusNet, self).__init__()
        self.initial = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # Use less pooling to preserve features
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        # Block 1
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(128)))
        self.layer3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm3d(256)))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = smallFocusNet()

    input_tensor = torch.randn(4, 3, 8, 512, 512)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    print("Output Tensor:", output.shape)
