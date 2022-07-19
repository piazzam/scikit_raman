import torch
from torch.utils.data import Dataset

class PytorchDataset(Dataset):
    def __init__(self, x, y, user):
        super(PytorchDataset).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.user = user

    def __getitem__(self, item):
        spectrum = self.x[item]
        labels = self.y[item]
        user = self.user[item]
        return spectrum, labels, user

    def __len__(self):
        return len(self.x)