import json, time
import torch
torch.manual_seed(0)

from cached_data import CachedMultimodalDataset
from tabular_data import NUM_TABULAR_FEATURES
from models_cached import (
    ImagingOnlyClassifier, TabularOnlyClassifier, LateFusionEnsemble,
    EarlyFusionClassifier, BilinearFusionClassifier,
)
from train_fusion import train_binary_classifier, evaluate_binary

t_start = time.time()


def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)


train_ds = CachedMultimodalDataset("outputs/cache_train.npz")
val_ds = CachedMultimodalDataset("outputs/cache_val.npz")
test_ds = CachedMultimodalDataset("outputs/cache_test.npz")
log(f"cached data loaded: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

results = {}

log("training ImagingOnlyClassifier...")
img_model = ImagingOnlyClassifier()
img_model, _, _ = train_binary_classifier(img_model, train_ds, val_ds, epochs=30, batch_size=32)
results["imaging_only"] = evaluate_binary(img_model, test_ds)
log(f"  imaging_only    test AUC={results['imaging_only']['auc']:.3f}  acc={results['imaging_only']['acc']:.3f}")

log("training TabularOnlyClassifier...")
tab_model = TabularOnlyClassifier(NUM_TABULAR_FEATURES)
tab_model, _, _ = train_binary_classifier(tab_model, train_ds, val_ds, epochs=30, batch_size=32)
results["tabular_only"] = evaluate_binary(tab_model, test_ds)
log(f"  tabular_only    test AUC={results['tabular_only']['auc']:.3f}  acc={results['tabular_only']['acc']:.3f}")

log("evaluating LateFusionEnsemble (averages the two models above, no joint training)...")
late_model = LateFusionEnsemble(img_model, tab_model)
results["late_fusion_ensemble"] = evaluate_binary(late_model, test_ds)
log(f"  late_fusion     test AUC={results['late_fusion_ensemble']['auc']:.3f}  acc={results['late_fusion_ensemble']['acc']:.3f}")

log("training EarlyFusionClassifier...")
early_model = EarlyFusionClassifier(NUM_TABULAR_FEATURES)
early_model, _, early_hist = train_binary_classifier(early_model, train_ds, val_ds, epochs=40, batch_size=32)
results["early_fusion"] = evaluate_binary(early_model, test_ds)
results["early_fusion"]["val_history"] = early_hist
log(f"  early_fusion    test AUC={results['early_fusion']['auc']:.3f}  acc={results['early_fusion']['acc']:.3f}")

log("training BilinearFusionClassifier...")
bilinear_model = BilinearFusionClassifier(NUM_TABULAR_FEATURES)
bilinear_model, _, bilinear_hist = train_binary_classifier(bilinear_model, train_ds, val_ds, epochs=40, batch_size=32)
results["bilinear_fusion"] = evaluate_binary(bilinear_model, test_ds)
results["bilinear_fusion"]["val_history"] = bilinear_hist
log(f"  bilinear_fusion test AUC={results['bilinear_fusion']['auc']:.3f}  acc={results['bilinear_fusion']['acc']:.3f}")

results["_meta"] = {"total_wall_time_s": time.time() - t_start}
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)
log(f"DONE. total wall time {time.time()-t_start:.0f}s")
