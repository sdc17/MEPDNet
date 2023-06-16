import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)  # 16
        self.channel_gate = ChannelGate2d(in_ch)

    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2
        return x


class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = input_x * x

        return x


def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1, dilation=1)
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, d, e=None):
        d = self.up(d)
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        out = self.spa_cha_gate(out)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU()
    )
    return block


class SCSEUnet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(SCSEUnet, self).__init__()
        # Encode
        self.conv_encode1 = nn.Sequential(contracting_block(in_channels=in_channel, out_channels=32), SCSE(32))
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = nn.Sequential(contracting_block(in_channels=32, out_channels=64), SCSE(64))
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = nn.Sequential(contracting_block(in_channels=64, out_channels=128), SCSE(128))
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = nn.Sequential(contracting_block(in_channels=128, out_channels=256), SCSE(256))
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SCSE(512)
        )
        
        # Decode
        self.conv_decode4 = expansive_block(512, 256, 256)
        self.conv_decode3 = expansive_block(256, 128, 128)
        self.conv_decode2 = expansive_block(128, 64, 64)
        self.conv_decode1 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    def forward(self, x):
        # set_trace()
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        # Bottleneck
        bottleneck = self.bottleneck(encode_pool4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)

        out = self.final_layer(decode_block1)
        # out = torch.sigmoid(out)

        return out

# net = SCSEUnet(in_channel=3, out_channel=2)
# summary(net, (3,512,512), device='cpu')