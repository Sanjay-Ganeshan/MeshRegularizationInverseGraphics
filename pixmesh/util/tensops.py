import torch
import pyredner as pyr
import numpy as np

gpu = lambda x: torch.tensor(x, dtype=torch.float32, device=pyr.get_device())
cpu = lambda x: torch.tensor(x, dtype=torch.float32)
gpui = lambda x: torch.tensor(x, dtype=torch.int, device=pyr.get_device())
cpui = lambda x: torch.tensor(x, dtype=torch.int)

