#!/usr/bin/env bash
#
# Fresh install for the RTX 5070 Ti (Blackwell, sm_120, 16 GB) on Ubuntu 26.
#
# The 5070 Ti is the SAME Blackwell architecture (sm_120) as the RTX 5050 this app
# was built on, so the CUDA 12.8 PyTorch build is identical — older/non-cu128 wheels
# throw "no kernel image available for execution on the device" on Blackwell.
#
# Run from the project root:  bash setup.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Pick a Python the pinned wheels actually support -------------------------------
# torch 2.11.0+cu128 ships cp39–cp313 wheels. Ubuntu 26 may default to a newer Python
# (3.14+) that has no matching wheel yet, so prefer 3.12/3.13 if present.
PY=""
for c in python3.12 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no python3 found. Install 3.12:  sudo apt install python3.12 python3.12-venv" >&2
  exit 1
fi
echo "Using $($PY --version 2>&1) at $(command -v "$PY")"

ver="$("$PY" -c 'import sys; print("%d%02d" % sys.version_info[:2])')"
if [ "$ver" -gt 313 ]; then
  echo
  echo "WARNING: $($PY --version 2>&1) is newer than torch 2.11.0+cu128's wheels (cp39–cp313)."
  echo "If 'pip install torch' below fails with 'no matching distribution', install 3.12:"
  echo "    sudo apt install python3.12 python3.12-venv"
  echo "then re-run this script (it prefers python3.12 automatically)."
  echo
fi

# --- Virtualenv ---------------------------------------------------------------------
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# --- PyTorch first, from the CUDA 12.8 index (required for Blackwell sm_120) ---------
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# --- Everything else (cu128 torch above already satisfies the torch pins) ------------
# requirements.txt includes the ultralytics CLIP fork needed for SAM 3 text prompts.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# --- Sanity check: confirm the GPU is actually usable -------------------------------
python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "no GPU"
print(f"\nCUDA available: {ok}  |  device: {name}  |  torch {torch.__version__}")
if not ok:
    raise SystemExit("CUDA not available — check NVIDIA driver + that this is a cu128 torch build.")
PY

cat <<'MSG'

Setup complete.

Run the app:
    .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000

Then open http://localhost:8000 — drop a video in ./input/, type a prompt, go.

  • YOLOE-26 works out of the box (ungated, auto-downloads).
  • For SAM 3: request access at https://huggingface.co/facebook/sam3,
    download sam3.pt, and place it in this project root. The app picks it up
    automatically (no restart needed for the next job).
MSG
