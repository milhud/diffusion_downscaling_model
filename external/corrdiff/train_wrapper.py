"""Runs physicsnemo's corrdiff train.py with the system TensorFlow blocked.

The GEOSpyD module's TensorFlow 2.17.0 is ABI-incompatible with the newer
numpy/protobuf that torch/tensorboard require here (crashes with a
SystemError in a compiled extension on import). train.py only needs
`torch.utils.tensorboard.SummaryWriter`, which works fine via tensorboard's
own stub writer when real TensorFlow is unavailable.

Blocking is done via a `tensorflow` stub package in this venv's own
site-packages (venv/lib/python3.12/site-packages/tensorflow/__init__.py,
which shadows the broken system install since the venv's own site-packages
comes first on sys.path) that raises a clean ImportError. We deliberately do
NOT set `sys.modules["tensorflow"] = None` here to fake the same thing:
einops's backend auto-detection (used by physicsnemo's GroupNorm) checks
`"tensorflow" in sys.modules` without checking the value isn't None, so a
bare `None` sentinel makes it try `import tensorflow` again later -- which
Python then refuses with "ModuleNotFoundError: import of tensorflow halted;
None in sys.modules", crashing training. A real (stubbed) module that raises
ImportError on import gets cleanly removed from sys.modules by Python
afterward, so that later probe just sees tensorflow as absent, as intended.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = "train.py"
    # runpy would otherwise put this wrapper's directory on sys.path[0]
    # instead of train.py's, breaking train.py's local `datasets` import.
    sys.path.insert(0, os.getcwd())
    runpy.run_path("train.py", run_name="__main__")
