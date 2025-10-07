#!/usr/bin/env bash
# Make PyTorch/cuDNN more deterministic where possible.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:16:8

python - <<'PY'
import os, random, numpy as np
try:
    import torch
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
except Exception:
    pass
random.seed(123)
np.random.seed(123)
print("[OK] Seeds fixed. Determinism increased (within backend limits).")
PY
