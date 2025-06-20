__all__ = ["ImageDataset"]
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

from denoising_config import *
import numpy as np
import re
import os
from PIL import Image

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split(r'([0-9]+)',key)]
    return sorted(data,key=alphanum_key)



class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.all_images = sorted_alphanumeric(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.all_images[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform参数不能为None，需指定预处理方法")
        # todo 高斯噪声和椒盐噪声混合注入策略
        nosie_images = tensor_image + torch.randn_like(tensor_image) * NOISE_FACTOR
        nosie_images = torch.clamp(nosie_images, 0., 1.)
        return nosie_images, tensor_image

if __name__ == '__main__':
    images = os.listdir(IMG_PATH)
    print(sorted_alphanumeric(images))