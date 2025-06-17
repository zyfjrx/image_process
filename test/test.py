import torch
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.all_images = os.listdir(self.image_dir)
    def __len__(self):
        return len(self.all_images)
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.all_images[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            tensor_image = self.transform(image)
        else:
            raise ValueError("transform参数不能为None，需指定预处理方法")
        noise_factor = 0.5
        nosie_images = tensor_image + torch.randn_like(tensor_image) * noise_factor
        return nosie_images,tensor_image
transform = transforms.Compose([transforms.Resize((68,68)),transforms.ToTensor()])
dataset = ImageDataset("/Users/zhangyf/PycharmProjects/image_process/common/dataset",transform)
print(len(dataset))
print(dataset[0])