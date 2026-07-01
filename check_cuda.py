import sys
import torch

sys.exit(0 if torch.cuda.is_available() else 1)
