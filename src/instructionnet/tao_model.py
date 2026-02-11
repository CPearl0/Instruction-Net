import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class InstructionEncoder(nn.Module):
    def __init__(self, instruction_repr_dim,
                 type_vocab_size=157,
                 type_embed_dim=192,
                 reg_linear_out=192,
                 branch_linear_out=192,
                 mem_linear_out=192):
        super().__init__()

        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.reg_linear = nn.Linear(64, reg_linear_out)
        self.branch_linear = nn.Linear(32, branch_linear_out)
        self.mem_linear = nn.Linear(64, mem_linear_out)

        concat_dim = type_embed_dim + reg_linear_out + branch_linear_out + mem_linear_out
        self.inst_linear = nn.Linear(concat_dim, instruction_repr_dim)

        self.norm = nn.LayerNorm(instruction_repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type_feat = x[..., 0]
        reg_feat = x[..., 1:65].float()
        mem_feat = x[..., 65:129].float()
        branch_feat = x[..., 129:161].float()

        type_embed = self.type_embedding(type_feat)
        reg_repr = self.reg_linear(reg_feat)
        branch_repr = self.branch_linear(branch_feat)
        mem_repr = self.mem_linear(mem_feat)
        concat = torch.cat((type_embed, reg_repr, branch_repr, mem_repr), dim=-1)
        concat = F.silu(concat)
        inst_repr = self.inst_linear(concat)
        inst_repr = self.norm(inst_repr)
        return F.silu(inst_repr)


class MultiTaskOutputHead(nn.Module):
    """
    使用一个线性层预测所有输出：
    1. fetch_cycle: 回归任务（直接预测）
    2. exec_cycle: 回归任务（直接预测）
    3. branch_mispredict: 二分类（sigmoid）
    4. tlb_hit: 二分类（sigmoid）
    5. dcache_hit: 4分类（softmax: L1/L2/L3/主存）
    6. icache_hit: 二分类（sigmoid）

    输出维度 = 1 + 1 + 1 + 1 + 4 + 1 = 9
    """
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.out_linear = nn.Linear(input_dim, 9)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.silu(self.norm(x))
        out = self.out_linear(x)
        fetch_cycle = F.softplus(out[..., 0])
        exec_cycle = F.softplus(out[..., 1])
        branch_mispred_logits = out[..., 2]
        tlb_hit_logits = out[..., 3]
        icache_hit_logits = out[..., 4]
        dcache_hit_logits = out[..., 5:9]
        
        branch_mispred = F.sigmoid(branch_mispred_logits)
        tlb_hit = F.sigmoid(tlb_hit_logits)
        icache_hit = F.sigmoid(icache_hit_logits)
        dcache_hit = F.softmax(dcache_hit_logits, dim=-1)

        return {
            "fetch_cycle": fetch_cycle,
            "exec_cycle": exec_cycle,
            "branch_mispred": branch_mispred,
            "tlb_hit": tlb_hit,
            "icache_hit": icache_hit,
            "dcache_hit": dcache_hit,
            # 用于loss计算
            "branch_mispred_logits": branch_mispred_logits,
            "tlb_hit_logits": tlb_hit_logits,
            "icache_hit_logits": icache_hit_logits,
            "dcache_hit_logits": dcache_hit_logits,
        }


class MultiHeadSelfAttention(nn.Module):
    """
    每个指令只能关注前面128条指令（包括自身共129条）。
    """
    def __init__(self, embed_dim, num_heads=8, window_size=128):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.RoPE = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        QKV = self.qkv_proj(x)
        QKV = rearrange(QKV, "... l (h d_k c) -> c ... h l d_k", h=self.num_heads, c=3)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        Q = self.RoPE.rotate_queries_or_keys(Q)
        K = self.RoPE.rotate_queries_or_keys(K)
        seq_len = x.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=-self.window_size)
        mask = torch.tril(mask)
        out = F.scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "... h l d_v -> ... l (h d_v)")
        return self.out_proj(out)


class TAOModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.inst_encoder = InstructionEncoder(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim)
        self.output_head = MultiTaskOutputHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inst_repr = self.inst_encoder(x)
        attn_out = self.attn(inst_repr)
        return self.output_head(attn_out)
