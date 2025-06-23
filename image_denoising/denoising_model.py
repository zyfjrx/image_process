import torch.nn as nn
import torch


# 搭建模型
# todo 优化UNet、NNet++、TransformerNNet
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器
        self.t_conv1 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=2, stride=2)
        # 普通卷积层
        self.conv_out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print(f'conv1:{x.shape}')
        x = self.pool(x)
        # print(f'pool1:{x.shape}')
        x = torch.relu(self.conv2(x))
        # print(f'conv2:{x.shape}')
        x = self.pool(x)
        # print(f'pool2:{x.shape}')
        x = torch.relu(self.conv3(x))
        # print(f'conv3:{x.shape}')
        x = self.pool(x)
        # print(f'pool3:{x.shape}')
        x = torch.relu(self.t_conv1(x))
        # print(f't_conv1:{x.shape}')
        x = torch.relu(self.t_conv2(x))
        # print(f't_conv2:{x.shape}')
        x = torch.relu(self.t_conv3(x))
        # print(f't_conv3:{x.shape}')
        x = torch.sigmoid(self.conv_out(x))
        # print(f'conv_out:{x.shape}')
        return x
if __name__ == '__main__':
    input = torch.randn(5, 3, 68, 68)
    model = ConvDenoiser()
    output = model(input)
    print(output.shape)