import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class nMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mse_loss = F.mse_loss(x, y, reduction='mean')
        var = torch.var(y)
        nmse_loss = mse_loss / var
        return nmse_loss