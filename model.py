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

'''
Note: this model only supports equal-sized minibatches for training or inference
Reimplementing torch.nn modules below to demonstrate understanding of their function and allow customization
Some reimplementations are based on my work completing ARENA (Alignment Research Engineer Accelerator)
'''
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
        '''
        Update the residual stream to have zero mean and unit variance.
        '''
        mu = resid.mean(dim=-1, keepdim=True)
        sigma = (resid.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        resid -= mu
        resid /= sigma
        resid *= self.w
        resid += self.b
        return resid
# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]
    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__()
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))
    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to the attention scores by zeroing the probs (actually setting logits to be effectively -inf)
        This means that information from later tokens cannot be moved to earlier ones, so that the model cannot cheat
        by merely attending entirely to the token it is trying to predict. This allows for training data to be used more
        efficiently, by having the model make predictions for each token, not merely the final token.
        '''
        mask = t.ones_like(attn_scores)
        mask = mask.triu(1).bool()
        return t.masked_fill(attn_scores, mask, self.IGNORE)
    def forward(self, norm_resid: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        # QK-Circuit
        x = norm_resid
        '''
        Calculate the key tensor by matmuling the (batched) embeddings of each token in the 
        residual stream with the learned key weights tensor then adding the learned biases.
        Resulting key tensor will be of size (batch, position_K, n_heads, d_head). This can 
        also be thought of as matmuling each attention head's key matrix independently with 
        the residual stream, but doing a larger mult is computationally more efficient.
        ''' 
        K = einops.einsum(self.W_K, x, "n_heads d_model d_head, batch position_K d_model -> batch position_K n_heads d_head") + self.b_K
        '''
        Calculate the query tensor by matmuling the learned query weights tensor with the
        (batched) embeddings of each token in the residual stream, then adding the learned biases.
        Resulting query tensor will be of size (batch, position_Q, n_heads, d_head). This can also
        be thought of as matmuling each attention head's query matrix independently with
        the residual stream, but doing a larger mult is computationally more efficient.
        '''
        Q = einops.einsum(x, self.W_Q, "batch position_Q d_model, n_heads d_model d_head -> batch position_Q n_heads d_head") + self.b_Q
        '''
        Calculate the attention tensor by finding the pairwise dot products of the activations of the query
        and key tensors. This QK matrix can be thought of as the fundamental object of the QK-Circuit,
        with the Q and K matrices really just being intermediate steps in creating this larger matrix. 
        This design choice allows the attention mechanism to handle residual streams of varying lengths.
        This pairwise dot product is mathematically equivalent to matmuling Q by K-transpose. The shape
        of the resulting QK tensor is (batch, n_heads, position_Q, position_K), equivalent to n_batches batches of
        n_heads independent attention heads, which each have a QK-matrix of size seq-length by seq-length
        '''
        QKT = einops.einsum(Q, K, "batch position_Q n_heads d_head, batch position_K n_heads d_head -> batch n_heads position_Q position_K")
        '''
        Rescale the QK matrix and apply causal mask, then convert the attention scores to probabilities by softmaxing them
        '''
        QKT = QKT / np.sqrt(self.cfg.d_head)
        A = self.apply_causal_mask(QKT).softmax(dim=-1)
        # OV-Circuit
        '''
        Calculate the value-tensor by matmuling each embedding in the residual stream with the value matrix for each attention head.
        The QK circuit allows the model to specify "how much" it does an operation at each token, roughly "how important is token y 
        to the meaning of token x w.r.t. whatever function that head is implementing", while the OV circuit allows the model to specify
        what operation it does on the embedding of token y before adding it to the meaning of token x. It can be helpful to consider
        the O and V matrices as merely a low-rank factorization of the OV matrix, which says roughly "how do we change the residual stream
        for other tokens if token y says blah", and this is then scaled by the QK-matrix.
        '''
        V = einops.einsum(self.W_V, x, "n_heads d_model d_head, batch position d_model -> batch position n_heads d_head") + self.b_V
        z = einops.einsum(A, V, "batch n_heads position_Q position_K, batch position_K n_heads d_head -> batch position_Q n_heads d_head")
        '''
        Take the result of scaling the value-tensor by the attention scores and then matmul this by the output matrices of each head.
        This can be thought of as either projecting this value back onto the residual stream (and whatever other linear operation happens here),
        or, as I prefer, one can consider O and V to just be a low-rank factorization of the true fundamental object here, OV, which
        is then scaled by the attention scores before being added to the residual stream. These two interpretations are mathematically equivalent,
        it is just programatically preferable to write code that does the former rather than the latter.
        '''
        result = einops.einsum(z, self.W_O, "batch position_Q n_heads d_head, n_heads d_head d_model -> batch position_Q n_heads d_model")
        '''
        Now sum the effects of each independent attention head and add the out-bias to determine how the residual stream will be updated.
        '''
        attn_out = einops.einsum(result, "batch position n_heads d_model -> batch position d_model") + self.b_O
        return attn_out
# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
    
    def forward(self, norm_resid: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        '''
        Matmul represents the residual stream passing through the first nodes of the mlp layer, after gelu this is the value of the hidden layer of neurons
        Then the next matmul represents the stream passing through the hidden layer to the output layer, and the result is the activations of the output layer.
        MLPs can implement things like knowledge, being a key-value pair, etc. The same operation is done at every place in the residual stream.
        '''
        hidden = einops.einsum(norm_resid, self.W_in, "batch position d_model, d_model d_mlp -> batch position d_mlp") + self.b_in
        gelu = nn.GELU(approximate='tanh')
        hidden = gelu(hidden)
        mlp_out = einops.einsum(hidden, self.W_out, "batch position d_mlp, d_mlp d_model -> batch position d_model") + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_model"]:
        resid += self.attn(self.ln1(resid))
        resid += self.mlp(self.ln2(resid))
        return resid
# %%
class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros(cfg.d_vocab))
    def forward(self, norm_resid_final: Float[Tensor, "batch position d_model"]) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(norm_resid_final, self.W_U, "batch position d_model, d_model d_vocab -> batch position d_vocab") + self.b_U
# %%
# %%
'''Tests'''
rand_input_test(Embed, [2, 4], float=False)
rand_input_test(PosEmbed, [2, 4], float=False)
rand_input_test(LayerNorm, [2, 4, 256])
rand_input_test(Attention, [2, 4, 256])
rand_input_test(MLP, [2, 4, 256])
rand_input_test(Unembed, [2, 4, 256])
# %%
