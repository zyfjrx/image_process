import torch
import numpy as np
import random
import os

# 定义函数：对所有模块都使用统一的随机种子
def seed_everything(seed):
    random.seed(seed)  # python内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed) # python哈希种子
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu随机种子
    torch.cuda.manual_seed(seed) # GPU随机种子
    torch.backends.cudnn.deterministic = True # 确保cudnn的确定性
    torch.backends.cudnn.benchmark = True # 启用cudnn性能优化