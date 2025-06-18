import torch.nn as nn
import torch

class Classification(nn.Module):
    def __init__(self,n_classes=5):
        super(Classification, self).__init__()
        self.conv1 = nn.Conv2d(3, 8,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(8,16,kernel_size=3,padding=1,stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.linear = nn.Linear(16**3,n_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    input = torch.randn(10,3,64,64)
    model = Classification()
    output = model(input)
    print(output.shape)