"""Print library versions and check that PyTorch can use the Apple GPU (MPS)."""
import platform
import sys

import matplotlib
import numpy as np
import torch

print(f"Python      {sys.version.split()[0]}")
print(f"Platform    {platform.platform()}")
print(f"torch       {torch.__version__}")
print(f"numpy       {np.__version__}")
print(f"matplotlib  {matplotlib.__version__}")

assert sys.version_info >= (3, 11), "Python 3.11+ is required"

mps = torch.backends.mps.is_available()
print(f"\nMPS built:     {torch.backends.mps.is_built()}")
print(f"MPS available: {mps}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.ones(3, device=device)
print(f"Tensor on {device}: {(x * 2).tolist()}")

if not mps:
    print("\nWARNING: MPS is not available, falling back to CPU.")
    sys.exit(1)
