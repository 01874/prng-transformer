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