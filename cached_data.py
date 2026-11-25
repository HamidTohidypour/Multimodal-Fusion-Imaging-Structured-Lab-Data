import numpy as np
import torch
from torch.utils.data import Dataset


class CachedMultimodalDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.img_embed = torch.from_numpy(d["img_embed"]).float()
        self.tab = torch.from_numpy(d["tab"]).float()
        self.y = torch.from_numpy(d["y"]).long()
        self.aux = torch.stack([torch.from_numpy(d["b_img"]).long(), torch.from_numpy(d["b_lab"]).long()], dim=1)

    def __len__(self):
        return self.img_embed.shape[0]

    def __getitem__(self, i):
        return self.img_embed[i], self.tab[i], self.y[i], self.aux[i]
