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
                 same_hist_linear_out=192):
        super().__init__()

        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.reg_linear = nn.Linear(64, reg_linear_out)
        self.branch_linear = nn.Linear(32, branch_linear_out)
        self.same_hist_linear = nn.Linear(192, same_hist_linear_out)

        concat_dim = type_embed_dim + reg_linear_out + branch_linear_out + same_hist_linear_out
        self.inst_linear = nn.Linear(concat_dim, instruction_repr_dim)

        self.norm = nn.LayerNorm(instruction_repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type_feat = x[..., 0]
        reg_feat = x[..., 1:65].float()
        same_hist_feat = x[..., 65:257].float()
        branch_feat = x[..., 257:289].float()

        type_embed = self.type_embedding(type_feat)
        reg_repr = self.reg_linear(reg_feat)
        branch_repr = self.branch_linear(branch_feat)
        same_hist_repr = self.same_hist_linear(same_hist_feat)
        concat = torch.cat((type_embed, reg_repr, branch_repr, same_hist_repr), dim=-1)
        concat = F.silu(concat)
        inst_repr = self.inst_linear(concat)
        inst_repr = self.norm(inst_repr)
        return F.silu(inst_repr)


class MultiTaskOutputHead(nn.Module):
    """
    Output heads for all tasks:
    1. fetch_cycle_class: 11 classes (class 0-9 for cycles 1-10, class 10 for 10+)
    2. fetch_cycle_regression: regression (used when cycle >= 11)
    3. exec_cycle_class: 11 classes
    4. exec_cycle_regression: regression (used when cycle >= 11)
    5. branch_mispredict: binary classification (sigmoid)
    6. icache_hit: 3 classes (softmax: L1/L2/Memory)
    7. dcache_hit: 3 classes (softmax: L1/L2/Memory)

    Output dim = 11 + 1 + 11 + 1 + 1 + 3 + 3 = 31
    """
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.out_linear = nn.Linear(input_dim, 31)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.silu(self.norm(x))
        out = self.out_linear(x)

        # fetch_cycle prediction: classification + regression
        fetch_cycle_class_logits = out[..., 0:11]
        fetch_cycle_regression = F.softplus(out[..., 11])

        # exec_cycle prediction: classification + regression
        exec_cycle_class_logits = out[..., 12:23]
        exec_cycle_regression = F.softplus(out[..., 23])

        # Other tasks
        branch_mispred_logits = out[..., 24]
        icache_hit_logits = out[..., 25:28]
        dcache_hit_logits = out[..., 28:31]

        branch_mispred = F.sigmoid(branch_mispred_logits)
        icache_hit = F.softmax(icache_hit_logits, dim=-1)
        dcache_hit = F.softmax(dcache_hit_logits, dim=-1)

        # Compute final fetch_cycle prediction
        fetch_cycle_class_pred = fetch_cycle_class_logits.argmax(dim=-1)
        fetch_cycle = torch.where(
            fetch_cycle_class_pred < 10,
            (fetch_cycle_class_pred + 1).float(),
            fetch_cycle_regression
        )

        # Compute final exec_cycle prediction
        exec_cycle_class_pred = exec_cycle_class_logits.argmax(dim=-1)
        exec_cycle = torch.where(
            exec_cycle_class_pred < 10,
            (exec_cycle_class_pred + 1).float(),
            exec_cycle_regression
        )

        return {
            "fetch_cycle": fetch_cycle,
            "fetch_cycle_class_logits": fetch_cycle_class_logits,
            "fetch_cycle_regression": fetch_cycle_regression,
            "exec_cycle": exec_cycle,
            "exec_cycle_class_logits": exec_cycle_class_logits,
            "exec_cycle_regression": exec_cycle_regression,
            "branch_mispred": branch_mispred,
            "icache_hit": icache_hit,
            "dcache_hit": dcache_hit,
            # For loss computation
            "branch_mispred_logits": branch_mispred_logits,
            "icache_hit_logits": icache_hit_logits,
            "dcache_hit_logits": dcache_hit_logits,
        }


class MultiHeadSelfAttention(nn.Module):
    """
    每个指令只能关注前面128条指令（包括自身共129条）。
    """
    def __init__(self, embed_dim, num_heads, positional_encoder: RotaryEmbedding, window_size=128):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.RoPE = positional_encoder

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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_encoder: RotaryEmbedding):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, positional_encoder)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = x + self.attn(self.norm1(x))
        ffn_out = attn_out + self.ffn(self.norm2(attn_out))
        return ffn_out


class InstructionNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.inst_encoder = InstructionEncoder(hidden_dim)
        self.RoPE = RotaryEmbedding(hidden_dim // 8)
        self.layers = nn.Sequential(
            *[TransformerBlock(hidden_dim, 8, hidden_dim * 8 // 3, self.RoPE)
              for _ in range(3)]
        )
        self.output_head = MultiTaskOutputHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.inst_encoder(x)
        x = self.layers(x)
        return self.output_head(x)
