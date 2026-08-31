#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-third_party/MoGe}"
MOGE_REVISION="${MOGE_REVISION:-7807b5d}"
UTILS3D_REVISION="c5daf6f6c244d251f252102d09e9b7bcef791a38"

# MoGe is optional and must not replace the backend's tested PyTorch build.
python -c 'import torch; print("Using existing PyTorch:", torch.__version__)'

if [[ ! -e "$TARGET_DIR" ]]; then
  git clone https://github.com/microsoft/MoGe.git "$TARGET_DIR"
elif [[ ! -d "$TARGET_DIR/.git" ]]; then
  echo "Existing MoGe target is not a git checkout: $TARGET_DIR" >&2
  exit 1
fi

if ! git -C "$TARGET_DIR" diff --quiet || ! git -C "$TARGET_DIR" diff --cached --quiet; then
  echo "MoGe checkout has local changes; refusing to change its revision." >&2
  exit 1
fi

git -C "$TARGET_DIR" checkout --detach "$MOGE_REVISION"

# Runtime-only dependencies for MoGe-2. --no-deps prevents any command here
# from upgrading torch, torchvision, CUDA, or NCCL.
python -m pip install --no-deps \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_REVISION}"
python -m pip install scipy opencv-python-headless plyfile moderngl

python -c "import sys, torch; sys.path.insert(0, '$TARGET_DIR'); from moge.model import import_model_class_by_version; import_model_class_by_version('v2'); print('MoGe-2 ready with PyTorch', torch.__version__)"
echo "MoGe-2 installed from: $TARGET_DIR@$MOGE_REVISION"
