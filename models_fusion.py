"""
Unimodal and fusion architectures for the imaging + tabular prediction task.

Four models, deliberately spanning the space of "how much does the fusion
mechanism matter":
  1. ImagingOnlyClassifier   - frozen SSL encoder embedding -> MLP -> logit
  2. TabularOnlyClassifier   - tabular MLP encoder -> MLP -> logit
  3. LateFusionEnsemble      - NOT TRAINED JOINTLY: averages the sigmoid
                               outputs of independently-trained (1) and (2).
                               This is a common "just ensemble the models we
                               already have" pattern in practice.
  4. EarlyFusionClassifier   - concatenates image + tabular embeddings BEFORE
                               the final nonlinearity, so the classifier head
                               can represent interaction terms.
  5. BilinearFusionClassifier - like (4) but explicitly appends the
                               elementwise product of the two embeddings as
                               additional input features, giving the head an
                               explicit interaction term rather than making it
                               discover one via depth.
"""
import torch
import torch.nn as nn

from models import Encoder3D


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


def load_frozen_image_encoder(ckpt_path, base=16, embed_dim=128):
    enc = Encoder3D(base=base, embed_dim=embed_dim)
    enc.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class ImagingOnlyClassifier(nn.Module):
    def __init__(self, image_encoder, img_embed_dim=128, hidden=32):
        super().__init__()
        self.image_encoder = image_encoder
        self.head = nn.Sequential(nn.Linear(img_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vol, tab):
        with torch.no_grad():
            img_embed = self.image_encoder(vol)
        return self.head(img_embed).squeeze(-1)


class TabularOnlyClassifier(nn.Module):
    def __init__(self, in_dim, tab_embed_dim=32, hidden=32):
        super().__init__()
        self.tab_encoder = TabularEncoder(in_dim, embed_dim=tab_embed_dim)
        self.head = nn.Sequential(nn.Linear(tab_embed_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, vol, tab):
        tab_embed = self.tab_encoder(tab)
        return self.head(tab_embed).squeeze(-1)


class LateFusionEnsemble(nn.Module):
    """Averages sigmoid outputs of two SEPARATELY trained unimodal models.
    No joint parameters -- this cannot represent interaction terms."""

    def __init__(self, imaging_model, tabular_model):
        super().__init__()
        self.imaging_model = imaging_model
        self.tabular_model = tabular_model

    def forward(self, vol, tab):
        with torch.no_grad():
            p_img = torch.sigmoid(self.imaging_model(vol, tab))
            p_tab = torch.sigmoid(self.tabular_model(vol, tab))
        p_avg = 0.5 * p_img + 0.5 * p_tab
        eps = 1e-6
        p_avg = p_avg.clamp(eps, 1 - eps)
        return torch.log(p_avg / (1 - p_avg))  # return as a logit for a consistent loss interface


class EarlyFusionClassifier(nn.Module):
    """Concatenates image + tabular embeddings before the classifier head.
    A big-enough MLP head can in principle discover interaction terms."""

    def __init__(self, image_encoder, tab_in_dim, img_embed_dim=128, tab_embed_dim=32, hidden=64):
        super().__init__()
        self.image_encoder = image_encoder
        self.tab_encoder = TabularEncoder(tab_in_dim, embed_dim=tab_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + tab_embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, vol, tab):
        with torch.no_grad():
            img_embed = self.image_encoder(vol)
        tab_embed = self.tab_encoder(tab)
        fused = torch.cat([img_embed, tab_embed], dim=1)
        return self.head(fused).squeeze(-1)


class BilinearFusionClassifier(nn.Module):
    """Early fusion + an explicit elementwise-product interaction feature,
    computed in a shared low-rank space so the model gets a direct interaction
    signal rather than having to discover one purely via MLP depth."""

    def __init__(self, image_encoder, tab_in_dim, img_embed_dim=128, tab_embed_dim=32,
                 interaction_dim=16, hidden=64):
        super().__init__()
        self.image_encoder = image_encoder
        self.tab_encoder = TabularEncoder(tab_in_dim, embed_dim=tab_embed_dim)
        self.img_proj = nn.Linear(img_embed_dim, interaction_dim)
        self.tab_proj = nn.Linear(tab_embed_dim, interaction_dim)
        self.head = nn.Sequential(
            nn.Linear(img_embed_dim + tab_embed_dim + interaction_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, vol, tab):
        with torch.no_grad():
            img_embed = self.image_encoder(vol)
        tab_embed = self.tab_encoder(tab)
        interaction = self.img_proj(img_embed) * self.tab_proj(tab_embed)
        fused = torch.cat([img_embed, tab_embed, interaction], dim=1)
        return self.head(fused).squeeze(-1)
