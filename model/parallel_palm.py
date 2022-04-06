import colossalai
import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import gather_forward_split_backward

from .palm import SwiGLU


class ParallelPalmTransformerLayer(nn.Module):

    def __init__(self, dim: int, dim_head: int=64, num_heads: int=8, ffn_mult: int=4, multi_query: bool=False):
        """
        """

        super().__init__()
        self.world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.attn_inner_dim = num_heads * dim_head
        self.ffn_inner_dim = int(ffn_mult * dim)

        # TODO: this currently only applies to 1D
        self.num_heads_per_partition = self.num_heads // self.world_size
        self.attn_inner_dim_per_partition = self.attn_inner_dim // self.world_size
        self.dim_head_per_partition = self.dim_head // self.world_size
        self.ffn_inner_dim_per_partition = self.ffn_inner_dim // self.world_size

        self.sanity_checks()
        
        # build the 2 fused linear layers
        self.multi_query = multi_query

        # calculate the projection size
        if self.multi_query:
            # only query has multi head
            # key and value remain as single head
            input_linear_dim = self.ffn_inner_dim * 2 + dim_head * (num_heads + 2)
        else:
            # conventional multi-head attention
            input_linear_dim = self.ffn_inner_dim * 2 + dim_head * num_heads * 3

        self.fused_input_linear = colossalai.nn.Linear(dim, input_linear_dim, bias=False)
        self.fused_output_linear = colossalai.nn.Linear(self.ffn_inner_dim + dim, dim, bias=False)

        # TODO: add rotary embedding
        # self.rotary_emb = ParallelRotaryEmbedding(self.dim_head)

        self.swiglu = SwiGLU()
        self.norm = colossalai.nn.LayerNorm(dim)
        self.scale = dim_head ** -0.5

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)


    def sanity_checks(self):
        # checks
        def _assert_divisible(dim, name):
            assert dim % self.world_size == 0, \
            f'{name} ({self.attn_inner_dim}) must be divisble by the world size ({self.world_size})'

        _assert_divisible(self.attn_inner_dim, 'Attention inner dimension')
        _assert_divisible(self.attn_inner_dim, 'Attention inner dimension')
        _assert_divisible(self.dim_head, 'Attention head dimension')
        _assert_divisible(self.ffn_inner_dim, 'FFN inner dimension')

    
    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        raise NotImplemented('RoPE Embedding has not been implementated yet')
    
    def forward(self, x):
        seq_length, device = x.shape[1], x.device
        # pre-layernorm
        x = self.norm(x)

        # fused input linear layer
        res_pack = self.fused_input_linear(x)

        if self.multi_query:
            q = res_pack.narrow(dim=2, 
                                start=0, 
                                length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(dim=2, 
                                start=self.attn_inner_dim_per_partition, 
                                length=self.dim_head_per_partition)
            v = res_pack.narrow(dim=2, 
                                start=(self.attn_inner_dim_per_partition + self.dim_head_per_partition), 
                                length=self.dim_head_per_partition)
            k = gather_forward_split_backward(k.contiguous(), parallel_mode=ParallelMode.TENSOR, dim=-1)
            v = gather_forward_split_backward(v.contiguous(), parallel_mode=ParallelMode.TENSOR, dim=-1)
            ffn_input = res_pack.narrow(dim=2,
                                        start=(self.attn_inner_dim_per_partition + 2 * self.dim_head_per_partition),
                                        length=self.ffn_inner_dim_per_partition)
        else:
            q = res_pack.narrow(dim=2, 
                                start=0, 
                                length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(dim=2, 
                                start=self.attn_inner_dim_per_partition, 
                                length=self.attn_inner_dim_per_partition)
            v = res_pack.narrow(dim=2, 
                                start=self.attn_inner_dim_per_partition * 2, 
                                length=self.attn_inner_dim_per_partition)
            ffn_input = res_pack.narrow(dim=2,
                                        start=self.attn_inner_dim_per_partition * 3,
                                        length=self.ffn_inner_dim_per_partition)
        
        # arrange the attention embeddings by head
        if not self.multi_query:
            k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads_per_partition)
            v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads_per_partition)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads_per_partition)

        # TODO: apply posititon embedding
        # q, k = apply_pos_embedding()

        # apply scale
        q = q * self.scale

        # calculate similarity
        if self.multi_query:
            sim = einsum("b h s d, b j d -> b h s j", q, k)
        else:
            # s and n here refer to sequence length
            # n is used only because einsum cannot have 2 same notations
            sim = einsum("b h s d, b h n d -> b h s n", q, k)

        # apply casual mask
        causal_mask = self.get_mask(seq_length, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        if self.multi_query:
            attn_out = einsum("b h i j, b j d -> b h i d", attn, v)
        else:
            attn_out = einsum("b h s n, b h n d -> b h s d", attn, v)

        # merge heads
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        concat_input = torch.cat([attn_out, ffn_input], dim=-1)
        out = self.fused_output_linear(concat_input)
        return out
        




