from torch.utils.data import Dataset

class XiguaData(Dataset):
    def __init__(self, type):
        if type == 'user':
