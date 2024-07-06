# %%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
import sys
import einops
from dataclasses import dataclass
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict, Callable
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
import wandb

device = t.device("cuda" if t.cuda.is_available() else "cpu")
#%%
# dataclass to store model hyperparams
@dataclass
class Config:
    d_model: int = 256
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 2
    init_range: float = 0.02
    n_ctx: int = 2048
    d_head: int = 32
    d_mlp: int = 768
    n_heads: int = 8
    n_layers: int = 8
# %%