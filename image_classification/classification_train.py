import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T  # 图像转换
from torch.utils.data import Dataset, DataLoader, random_split  # 数据集和数据加载器

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条工具

from classification_data import *
from classification_model import *
from classification_config import *
from classification_engine import *
from common import utils

if __name__ == '__main__':
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(SEED)
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    print("----------1.创建数据集--------")
    dataset = ImageLabelDataset(image_dir=IMG_PATH,label_path=LABELS_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])
    print("----------创建数据集完成--------")
    print("----------2.创建数据加载器--------")
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("--- 创建数据加载器完成 ---")

    model = Classification().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    min_test_loss = float('inf')
    print("----------3.开始训练模型 --------")
    train_loss_list = []
    test_loss_list = []
    test_correct_list = []
    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)
        train_loss_list.append(train_loss)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")

        test_loss, test_correct_num = evl_epoch(model, loss_fn, test_loader, device)
        test_loss_list.append(test_loss)
        accuracy = test_correct_num / len(test_dataset)
        test_correct_list.append(accuracy)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.6f}")

        if test_loss < min_test_loss:
            print("测试误差减小了，保存模型 ...")
            min_test_loss = test_loss
            torch.save(model.state_dict(), CLASSIFIER_MODEL_NAME)
        else:
            print("测试误差没有减小，不做保存！")
    print("----------训练完成---------")
    # loss 曲线
    plt.plot(train_loss_list, 'r--', label='train loss')
    plt.plot(test_loss_list, 'k--', label='test loss')
    plt.plot(test_correct_list, 'b--', label='test acc')
    plt.legend(loc='best')
    plt.show()

    loaded_model = Classification()
    loaded_model.load_state_dict(torch.load(CLASSIFIER_MODEL_NAME,map_location=device))
    loaded_model.eval().to(device)
    correct_count = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = loaded_model(x)
            pred = out.argmax(dim=1)
            correct_count += pred.eq(y).sum().item()
    print(f"模型测试准确率为：{correct_count / len(test_dataset)}")

