import torch
import torch.nn as nn

# 定义编码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        return x

# 定义解码器类
class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.conv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv3 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.conv4 = nn.ConvTranspose2d(32,16,kernel_size=2,stride=2)
        self.conv5 = nn.ConvTranspose2d(16,3,kernel_size=2,stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = torch.relu(self.conv3(x))
        print(x.shape)
        x = torch.relu(self.conv4(x))
        print(x.shape)
        x = torch.relu(self.conv5(x))
        return x

if __name__ == '__main__':
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    input = torch.randn(5, 3, 64, 64)
    output = encoder(input)
    y = decoder(output)
    print(y.shape)