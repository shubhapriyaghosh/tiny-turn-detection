import pandas as pd
import torch
from torch.utils.data import Dataset
from src.features import extract


class TurnDetectionDataset(Dataset):
    def __init__(self, manifest_path, feature_cfg, augment=None):
        self.df = pd.read_csv(manifest_path)
        self.feature_cfg = feature_cfg
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["audio_path"]
        if self.augment is not None:
            path = self.augment(path)
        mel, prosody = extract(path, self.feature_cfg)
        return torch.from_numpy(mel).unsqueeze(0), torch.from_numpy(prosody), torch.tensor(int(row["label"]), dtype=torch.long)


def collate_fn(batch):
    mels, prosodies, labels = zip(*batch)
    max_t = max(m.shape[-1] for m in mels)
    padded = [torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) if m.shape[-1] < max_t else m for m in mels]
    return torch.stack(padded), torch.stack(prosodies), torch.stack(labels)
