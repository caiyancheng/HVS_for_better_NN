import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # 当前 GPU
    torch.cuda.manual_seed_all(seed)  # 所有 GPU（多卡）

    torch.backends.cudnn.deterministic = True  # 保证每次卷积结果一样（可能稍慢）
    torch.backends.cudnn.benchmark = False     # 关闭自动优化卷积算法选择（可复现）