import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def train_binary_classifier(model, train_ds, val_ds, epochs=20, batch_size=32, lr=1e-3, verbose=False, trainable_only=True):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad] if trainable_only else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4) if params else None
    crit = nn.BCEWithLogitsLoss()

    best_auc, best_state = -1, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        if opt is not None:
            model.train()
            for img_embed, tab, y, aux in train_loader:
                logits = model(img_embed, tab)
                loss = crit(logits, y.float())
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for img_embed, tab, y, aux in val_loader:
                logits = model(img_embed, tab)
                all_logits.append(logits)
                all_y.append(y)
        all_logits = torch.cat(all_logits).numpy()
        all_y = torch.cat(all_y).numpy()
        try:
            val_auc = roc_auc_score(all_y, all_logits)
        except ValueError:
            val_auc = 0.5
        history.append({"epoch": epoch + 1, "val_auc": val_auc})
        if val_auc >= best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {epoch+1}/{epochs}  val_auc={val_auc:.3f}")
        if opt is None:
            break  # no trainable params (e.g. late-fusion ensemble) -- nothing to iterate on

    model.load_state_dict(best_state)
    return model, best_auc, history


@torch.no_grad()
def evaluate_binary(model, test_ds, batch_size=64):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_logits, all_y, all_bimg, all_blab = [], [], [], []
    for img_embed, tab, y, aux in loader:
        logits = model(img_embed, tab)
        all_logits.append(logits)
        all_y.append(y)
        all_bimg.append(aux[:, 0])
        all_blab.append(aux[:, 1])
    import torch as T
    logits = T.cat(all_logits).numpy()
    y = T.cat(all_y).numpy()
    b_img = T.cat(all_bimg).numpy()
    b_lab = T.cat(all_blab).numpy()
    preds = (logits > 0).astype(int)
    auc = roc_auc_score(y, logits) if len(set(y.tolist())) > 1 else 0.5
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    return {
        "auc": auc, "acc": acc, "f1": f1,
        "logits": logits.tolist(), "y": y.tolist(), "preds": preds.tolist(),
        "b_img": b_img.tolist(), "b_lab": b_lab.tolist(),
    }
