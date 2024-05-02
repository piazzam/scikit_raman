import torch
from torch.utils.data import Dataset

class PytorchDataset(Dataset):
    """
        A class that represent a dataset in Pytorch.
        ...

        Attributes
        ----------
        x : np.array
            spectras of the dataset.
        y: np.array
            labels of the dataset.
        user: np.array
            user names of the dataset.
        """
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