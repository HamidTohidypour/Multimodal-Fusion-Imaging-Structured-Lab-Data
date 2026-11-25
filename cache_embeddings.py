import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from tabular_data import make_multimodal_splits
from models_fusion import load_frozen_image_encoder

t0 = time.time()
train_ds, val_ds, test_ds = make_multimodal_splits(n_train=900, n_val=300, n_test=400, seed=0, label_noise_rate=0.05)
encoder = load_frozen_image_encoder("/home/claude/prenuvo_multimodal/checkpoints/ssl_encoder_v1.pt")
print(f"data + encoder ready ({time.time()-t0:.1f}s)", flush=True)


def cache_split(ds, name):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    img_embeds, tabs, ys, b_imgs, b_labs = [], [], [], [], []
    with torch.no_grad():
        for vol, tab, y, aux in loader:
            embed = encoder(vol)
            img_embeds.append(embed.numpy())
            tabs.append(tab.numpy())
            ys.append(np.array(y))
            b_imgs.append(aux[:, 0].numpy())
            b_labs.append(aux[:, 1].numpy())
    out = dict(
        img_embed=np.concatenate(img_embeds, 0),
        tab=np.concatenate(tabs, 0),
        y=np.concatenate(ys, 0),
        b_img=np.concatenate(b_imgs, 0),
        b_lab=np.concatenate(b_labs, 0),
    )
    np.savez(f"/home/claude/prenuvo_multimodal/outputs/cache_{name}.npz", **out)
    print(f"  cached {name}: {out['img_embed'].shape[0]} samples ({time.time()-t0:.1f}s elapsed)", flush=True)


cache_split(train_ds, "train")
cache_split(val_ds, "val")
cache_split(test_ds, "test")
print(f"DONE caching in {time.time()-t0:.1f}s")
