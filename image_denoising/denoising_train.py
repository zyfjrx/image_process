import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T  # 图像转换
from torch.utils.data import Dataset, DataLoader, random_split  # 数据集和数据加载器

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条工具

# 导入自定义组件
from common import utils
from denoising_config import *
from denoising_data import *
from denoising_model import *
from denoising_test import test
from denoising_engine import train_epoch, evl_epoch

if __name__ == '__main__':
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(SEED)
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    print("----------1.创建数据集--------")
    dataset = ImageDataset(image_dir=IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])
    print("----------创建数据集完成--------")
    print("----------2.创建数据加载器--------")
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("--- 创建数据加载器完成 ---")

    model = ConvDenoiser().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    min_test_loss = float('inf')
    print("----------3.开始训练模型 --------")
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)
        train_loss_list.append(train_loss)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")

        test_loss = evl_epoch(model, loss_fn, test_loader, device)
        test_loss_list.append(test_loss)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Test Loss: {test_loss}")

        if test_loss < min_test_loss:
            print("测试误差减小了，保存模型 ...")
            min_test_loss = test_loss
            torch.save(model.state_dict(),DENOISER_MODEL_NAME)
        else:
            print("测试误差没有减小，不做保存！")
    print("----------训练完成---------")
    # loss 曲线
    plt.plot(train_loss_list,'r--',label='train loss')
    plt.plot(test_loss_list,'k--',label='test loss')
    plt.legend(loc='best')
    plt.show()


    print("----------4.从文件加载模型---------")
    loaded_model = ConvDenoiser()
    loaded_model.load_state_dict(torch.load(DENOISER_MODEL_NAME,map_location=device))
    loaded_model.eval().to(device)
    print("----------加载模型完成---------")
    test(loaded_model,test_loader,device)
    print("----------测试完成---------")



