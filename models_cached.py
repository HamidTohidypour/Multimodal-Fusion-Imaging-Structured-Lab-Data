"""
Same architectures as models_fusion.py, but consuming a precomputed frozen
image embedding directly (forward(img_embed, tab)) instead of a raw volume +
encoder. Since the image encoder is frozen and never updated, its output is
fixed for a given volume -- caching it once (cache_embeddings.py) and reusing
it removes the CNN forward pass from every training epoch, which is the
dominant cost on CPU. This has no effect on what any model can represent;
it's purely a compute optimization enabled by the encoder being frozen.
"""
import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ImagingOnlyClassifier(nn.Module):
    def __init__(self, img_embed_dim=128, hidden=32):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(img_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, img_embed, tab):
        return self.head(img_embed).squeeze(-1)


class TabularOnlyClassifier(nn.Module):
    def __init__(self, in_dim, tab_embed_dim=32, hidden=32):
        super().__init__()
        self.tab_encoder = TabularEncoder(in_dim, embed_dim=tab_embed_dim)
        self.head = nn.Sequential(nn.Linear(tab_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, img_embed, tab):
        return self.head(self.tab_encoder(tab)).squeeze(-1)


class LateFusionEnsemble(nn.Module):
    """Averages sigmoid outputs of two SEPARATELY trained unimodal models.
    No joint parameters -- cannot represent interaction terms."""

    def __init__(self, imaging_model, tabular_model):
        super().__init__()
        self.imaging_model = imaging_model
        self.tabular_model = tabular_model

    def forward(self, img_embed, tab):
        with torch.no_grad():
            p_img = torch.sigmoid(self.imaging_model(img_embed, tab))
            p_tab = torch.sigmoid(self.tabular_model(img_embed, tab))
        p_avg = (0.5 * p_img + 0.5 * p_tab).clamp(1e-6, 1 - 1e-6)
        return torch.log(p_avg / (1 - p_avg))


class EarlyFusionClassifier(nn.Module):
    """Concatenates image + tabular embeddings before the classifier head."""

    def __init__(self, tab_in_dim, img_embed_dim=128, tab_embed_dim=32, hidden=64):
        super().__init__()
        self.tab_encoder = TabularEncoder(tab_in_dim, embed_dim=tab_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + tab_embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, img_embed, tab):
        tab_embed = self.tab_encoder(tab)
        fused = torch.cat([img_embed, tab_embed], dim=1)
        return self.head(fused).squeeze(-1)


class BilinearFusionClassifier(nn.Module):
    """Early fusion + an explicit elementwise-product interaction feature."""

    def __init__(self, tab_in_dim, img_embed_dim=128, tab_embed_dim=32, interaction_dim=16, hidden=64):
        super().__init__()
        self.tab_encoder = TabularEncoder(tab_in_dim, embed_dim=tab_embed_dim)
        self.img_proj = nn.Linear(img_embed_dim, interaction_dim)
        self.tab_proj = nn.Linear(tab_embed_dim, interaction_dim)
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + tab_embed_dim + interaction_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, img_embed, tab):
        tab_embed = self.tab_encoder(tab)
        interaction = self.img_proj(img_embed) * self.tab_proj(tab_embed)
        fused = torch.cat([img_embed, tab_embed, interaction], dim=1)
        return self.head(fused).squeeze(-1)
