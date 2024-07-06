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
    n_ctx: int = 1024
    d_head: int = 32
    d_mlp: int = 768
    n_heads: int = 8
    n_layers: int = 8
# %%
def rand_input_test(layer_class, shape, float: bool=True):
    cfg = Config(debug=True)
    layer = layer_class(cfg).to(device)
    if float:
        rand_input = t.randn(shape).to(device)
    else:
        rand_input = t.randint(0, cfg.d_vocab, shape).to(device)
    print("Input shape: ", rand_input.shape)
    out = layer(rand_input)
    if isinstance(out, tuple): out = out[0]
    print("Output shape: ", out.shape, "\n")

'''Note: this model only supports equal-sized minibatches for training or inference'''
# %%
class Embed(nn.Module):
    # let the model learn how it wants to embed the tokens
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        print(tokens.shape)
        return self.W_E[tokens]
# %%
class PosEmbed(nn.Module):
    # use absolute learned positional encoding, i.e. let the model learn how it wants to encode position.
    # might be interesting to look at how it encodes it in the future, and if this changes between generators
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos, std=cfg.init_range)
    
    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "position d_model -> batch position d_model", batch=batch)

# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(self, resid: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        mu = resid.mean(dim=-1, keepdim=True)
        sigma = (resid.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        resid -= mu
        resid /= sigma
        resid *= self.w
        resid += self.b
        return resid
# %%
'''Tests'''
rand_input_test(Embed, [2, 4], float=False)
rand_input_test(PosEmbed, [2, 4], float=False)
rand_input_test(LayerNorm, [2, 4, 256])
# %%
