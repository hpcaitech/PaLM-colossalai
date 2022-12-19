import colossalai
import torch
import torch.nn as nn
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import CheckpointModule
from einops import rearrange
from torch import dtype, einsum, matmul

from model.palm_utils import RotaryEmbedding, SwiGLU, apply_rotary_pos_emb
from model.parallel_utils import (
    gather_fwd_reduce_scatter_bwd,
    get_parallel_mode_for_gather,
    partition_by_tp,
)


class ParallelPalmTransformerLayer(CheckpointModule):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        num_heads: int = 8,
        ffn_mult: int = 4,
        multi_query: bool = False,
        use_grad_checkpoint=False,
        use_act_offload=False,
    ):
        """Fused PaLM transformer block with tensor parallelism"""

        super().__init__(checkpoint=use_grad_checkpoint, offload=use_act_offload)
        self.world_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.attn_inner_dim = num_heads * dim_head
        self.ffn_inner_dim = int(ffn_mult * dim)
        self.ffn_mult = ffn_mult

        self.num_heads_per_partition = partition_by_tp(self.num_heads)
        self.attn_inner_dim_per_partition = partition_by_tp(self.attn_inner_dim)
        self.dim_head_per_partition = partition_by_tp(self.dim_head)
        self.ffn_inner_dim_per_partition = partition_by_tp(self.ffn_inner_dim)

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
        self.mode_for_gahter = get_parallel_mode_for_gather()
        self.fused_output_linear = colossalai.nn.Linear(self.ffn_inner_dim + dim_head * num_heads, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.dim_head)
        self.swiglu = SwiGLU()
        self.norm = colossalai.nn.LayerNorm(dim, bias=False)
        self.scale = dim_head**-0.5

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, seq, device):
        if self.mask is not None and self.mask.shape[-1] >= seq:
            return self.mask[:seq, :seq]

        mask = torch.ones((seq, seq), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, seq, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq:
            return self.pos_emb[:seq]

        pos_emb = self.rotary_emb(seq, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def _forward(self, x):
        seq_length, device = x.shape[1], x.device
        # pre-layernorm
        x = self.norm(x)

        # fused input linear layer
        res_pack = self.fused_input_linear(x)

        if self.multi_query:
            q = res_pack.narrow(dim=2, start=0, length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(dim=2, start=self.attn_inner_dim_per_partition, length=self.dim_head_per_partition)
            v = res_pack.narrow(
                dim=2,
                start=(self.attn_inner_dim_per_partition + self.dim_head_per_partition),
                length=self.dim_head_per_partition,
            )
            if self.mode_for_gahter is not None:
                k = gather_fwd_reduce_scatter_bwd(k.contiguous(), parallel_mode=self.mode_for_gahter, dim=-1)
                v = gather_fwd_reduce_scatter_bwd(v.contiguous(), parallel_mode=self.mode_for_gahter, dim=-1)
            ffn_input = res_pack.narrow(
                dim=2,
                start=(self.attn_inner_dim_per_partition + 2 * self.dim_head_per_partition),
                length=self.ffn_inner_dim_per_partition * 2,
            )
        else:
            q = res_pack.narrow(dim=2, start=0, length=self.attn_inner_dim_per_partition)
            k = res_pack.narrow(
                dim=2, start=self.attn_inner_dim_per_partition, length=self.attn_inner_dim_per_partition
            )
            v = res_pack.narrow(
                dim=2, start=self.attn_inner_dim_per_partition * 2, length=self.attn_inner_dim_per_partition
            )
            ffn_input = res_pack.narrow(
                dim=2, start=self.attn_inner_dim_per_partition * 3, length=self.ffn_inner_dim_per_partition * 2
            )

        # arrange the attention embeddings by head
        if not self.multi_query:
            k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads_per_partition)
            v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads_per_partition)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads_per_partition)

        # apply posititon embedding
        positions = self.get_rotary_embedding(seq_length, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # apply scale
        q = q * self.scale

        # calculate similarity
        if self.multi_query:
            #sim = einsum("b h s d, b j d -> b h s j", q, k)
            sim = matmul(q, k.transpose(1,2))
        else:
            # s and n here refer to sequence length
            # n is used only because einsum cannot have 2 same notations
            #sim = einsum("b h s d, b h n d -> b h s n", q, k)
            sim = matmul(q, k.transpose(2,3))

        # apply casual mask
        causal_mask = self.get_mask(seq_length, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values
        if self.multi_query:
            #attn_out = einsum("b h i j, b j d -> b h i d", attn, v)
            attn_out = matmul(attn, v)
        else:
            #attn_out = einsum("b h s n, b h n d -> b h s d", attn, v)
            attn_out = matmul(attn, v)

        # merge heads
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        # mlp swiglu
        ffn_input = self.swiglu(ffn_input)

        concat_input = torch.cat([attn_out, ffn_input], dim=-1)
        out = self.fused_output_linear(concat_input)
        return out + x


class PaLMHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        word_embedding_weight: nn.Parameter = None,
        bias: bool = False,
        dtype: dtype = None,
    ) -> None:
        super().__init__()
        self.dense = colossalai.nn.Classifier(dim, num_tokens, word_embedding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x


def Parallel_PaLM(
    dim,
    num_tokens,
    depth,
    dim_head=64,
    num_heads=8,
    ff_mult=4,
    multi_query=False,
    use_grad_checkpoint=False,
    use_act_offload=False,
):
    word_embedding = colossalai.nn.Embedding(num_tokens, dim)
    net = nn.Sequential(
        word_embedding,
        *[
            ParallelPalmTransformerLayer(
                dim=dim,
                dim_head=dim_head,
                num_heads=num_heads,
                ffn_mult=ff_mult,
                multi_query=multi_query,
                use_grad_checkpoint=use_grad_checkpoint,
                use_act_offload=use_act_offload,
            )
            for _ in range(depth)
        ],
        colossalai.nn.LayerNorm(dim, bias=False),
        # they used embedding weight tied projection out to logits, not common, but works
        PaLMHead(dim=dim, num_tokens=num_tokens, word_embedding_weight=word_embedding.weight, bias=False),
    )

    nn.init.normal_(net[0].weight, std=0.02)
    return net
