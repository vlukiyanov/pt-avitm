import torch
from torch.utils.data import Dataset


class CountTensorDataset(Dataset):
    def __init__(self, matrix):
        self.matrix = matrix

    def __getitem__(self, index: int) -> torch.Tensor:
        return (torch.tensor(self.matrix.getrow(index).todense()).squeeze(0),)

    def __len__(self) -> int:
        return self.matrix.shape[0]
