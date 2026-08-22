import torch
from torch.utils.data import Dataset


class CachedTurnDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.mels = data["mels"]
        self.prosodies = data["prosodies"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], self.prosodies[idx], self.labels[idx]
