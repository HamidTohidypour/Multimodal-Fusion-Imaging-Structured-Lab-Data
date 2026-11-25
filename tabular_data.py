"""
Synthetic structured/lab data, paired with the same 3D volumes used in the
SSL project (same indices, same seed -> same shapes/classes).

Label design (deliberate, not arbitrary): the target is an INTERACTION-ONLY
signal between an image-derived attribute and a lab value:

    b_img = 1 if the volume's shape is "compact" (sphere/cube/cross)
            0 if "elongated" (ellipsoid/cylinder/ring)
    b_lab = 1 if a synthetic lab marker is above the population median
    y     = XOR(b_img, b_lab)      [plus a small label-noise flip rate]

This is constructed so that:
  - E[y | b_img] = 0.5 for every value of b_img (imaging carries ZERO
    marginal information about y)
  - E[y | b_lab] = 0.5 for every value of b_lab (labs carry ZERO marginal
    information about y)
  - E[y | b_img, b_lab] is fully determined (100% informative jointly)

So any unimodal model, or any model that only *combines* independently-formed
unimodal predictions (e.g. averaging two calibrated probabilities), is
mathematically capped at ~chance-level performance. Only a model that
represents both modalities jointly before its final nonlinearity can solve
this. This makes "does fusion actually help, and what kind of fusion" a
provable question rather than an anecdotal one.

Real clinical interactions are rarely a pure XOR, but many are: a lab marker
that's only prognostic in the presence of (or absence of) a particular
imaging finding is exactly this shape of problem, just noisier.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from data import generate_volume, NUM_CLASSES

COMPACT_CLASSES = {0, 1, 4}     # sphere, cube, cross
ELONGATED_CLASSES = {2, 3, 5}   # ellipsoid, cylinder, ring

TABULAR_FEATURE_NAMES = [
    "age", "bmi", "sex_female", "lab_marker_primary",
    "lab_marker_noise1", "lab_marker_noise2", "lab_marker_noise3",
]
NUM_TABULAR_FEATURES = len(TABULAR_FEATURE_NAMES)


def _make_tabular_row(index, seed, label_noise_rate=0.05):
    rng = np.random.default_rng(seed * 7_000_003 + index + 1)  # offset so it's independent of image rng
    _, img_class = generate_volume(index, size=24, seed=seed)  # only used to read off the class label
    b_img = 1 if img_class in COMPACT_CLASSES else 0

    age = rng.normal(55, 15)
    bmi = rng.normal(26, 4)
    sex_female = rng.integers(0, 2)

    lab_primary_raw = rng.normal(0, 1)          # this is the ONE feature that determines b_lab
    noise1 = rng.normal(0, 1)
    noise2 = rng.normal(0, 1)
    noise3 = rng.normal(0, 1)

    b_lab = 1 if lab_primary_raw > 0 else 0      # median split (population median ~ 0)

    y = int(b_img != b_lab)  # XOR
    if rng.uniform() < label_noise_rate:
        y = 1 - y  # label noise, as in any real annotation process

    features = np.array([age, bmi, sex_female, lab_primary_raw, noise1, noise2, noise3], dtype=np.float32)
    return features, y, b_img, b_lab, img_class


class MultimodalOrganDataset(Dataset):
    """Returns (volume, tabular_features, label, aux) for each sample.
    aux carries (b_img, b_lab, img_class) for diagnostic breakdowns."""

    def __init__(self, indices, size=24, seed=0, label_noise_rate=0.05, tabular_mean=None, tabular_std=None):
        self.indices = list(indices)
        self.size = size
        self.seed = seed
        self.label_noise_rate = label_noise_rate
        self.tabular_mean = tabular_mean
        self.tabular_std = tabular_std

    def __len__(self):
        return len(self.indices)

    def raw_tabular(self, i):
        return _make_tabular_row(self.indices[i], self.seed, self.label_noise_rate)

    def __getitem__(self, i):
        vol, img_class = generate_volume(self.indices[i], size=self.size, seed=self.seed)
        features, y, b_img, b_lab, _ = _make_tabular_row(self.indices[i], self.seed, self.label_noise_rate)
        if self.tabular_mean is not None:
            features = (features - self.tabular_mean) / self.tabular_std
        vol_t = torch.from_numpy(vol).unsqueeze(0)
        feat_t = torch.from_numpy(features.astype(np.float32))
        return vol_t, feat_t, y, torch.tensor([b_img, b_lab], dtype=torch.long)


def compute_tabular_normalization(indices, seed=0, label_noise_rate=0.05):
    """Fit mean/std on a set of indices (train split) for standardization."""
    rows = [_make_tabular_row(i, seed, label_noise_rate)[0] for i in indices]
    arr = np.stack(rows, axis=0)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-6


def make_multimodal_splits(n_train=900, n_val=300, n_test=400, size=24, seed=0, label_noise_rate=0.05):
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    mean, std = compute_tabular_normalization(train_idx, seed=seed, label_noise_rate=label_noise_rate)

    train_ds = MultimodalOrganDataset(train_idx, size=size, seed=seed, label_noise_rate=label_noise_rate,
                                       tabular_mean=mean, tabular_std=std)
    val_ds = MultimodalOrganDataset(val_idx, size=size, seed=seed, label_noise_rate=label_noise_rate,
                                     tabular_mean=mean, tabular_std=std)
    test_ds = MultimodalOrganDataset(test_idx, size=size, seed=seed, label_noise_rate=label_noise_rate,
                                      tabular_mean=mean, tabular_std=std)
    return train_ds, val_ds, test_ds
