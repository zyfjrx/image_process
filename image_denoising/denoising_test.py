import torch
import torch.nn as nn
import torchvision.transforms as T  # 图像转换
from torch.utils.data import Dataset, DataLoader, random_split    # 数据集和数据加载器

import numpy as np
import matplotlib.pyplot as plt

# 导入自定义组件
from common import utils
from denoising_config import *
from denoising_data import *
from denoising_model import *


def test(model, test_loader, device):
    model.eval()
    data_iter = iter(test_loader)
    noise_images, original_images = next(data_iter)
    noise_images = noise_images.to(device)
    original_images = original_images.to(device)
    output = model(noise_images)

    noise_images = noise_images.permute(0,2,3,1).cpu().numpy()
    original_images = original_images.permute(0,2,3,1).cpu().numpy()
    output = output.permute(0,2,3,1).detach().cpu().numpy()

    fig,axes = plt.subplots(3,10,figsize=(25,4))
    for images,row in zip([noise_images,output,original_images],axes):
        for image,ax in zip(images,row):
            ax.imshow(image)
            ax.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    utils.seed_everything(SEED)
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataset(image_dir=IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])
    test_loader = DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE)
    model = ConvDenoiser()
    model.load_state_dict(torch.load(DENOISER_MODEL_NAME,map_location=device))
    model.eval().to(device)
    test(model, test_loader, device)
