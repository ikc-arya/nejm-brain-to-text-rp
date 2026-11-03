#!/bin/bash
# Create the uv env from requirements.txt. Run from the prt-rpn-karya root.
set -euo pipefail
cd "$(dirname "$0")"
export UV_LINK_MODE=copy  # macOS rejects code signatures of clone-linked .so files (seen with h5py)
uv venv --clear --python 3.10 .venv-train
uv pip install --python .venv-train/bin/python -r requirements.txt
# put src/ on the path (modules use flat imports: `from rnn_model import GRUDecoder`)
echo "$PWD/src" > .venv-train/lib/python3.10/site-packages/prt_rpn_karya_src.pth
