import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class InstructionEncoder(nn.Module):
    def __init__(self, instruction_repr_dim,
                 type_vocab_size=157,
                 type_embed_dim=256,
                 reg_linear_out=192,
                 branch_linear_out=192,
                 icache_hist_linear_out=128,
                 dcache_hist_linear_out=128,
                 page_hist_linear_out=128,
                 flag_linear_out=32):
        super().__init__()

        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.reg_linear = nn.Linear(64, reg_linear_out)
        self.branch_linear = nn.Linear(32, branch_linear_out)
        self.icache_hist_linear = nn.Linear(64, icache_hist_linear_out)
        self.dcache_hist_linear = nn.Linear(64, dcache_hist_linear_out)
        self.page_hist_linear = nn.Linear(64, page_hist_linear_out)
        self.flag_linear = nn.Linear(3, flag_linear_out)

        concat_dim = type_embed_dim + reg_linear_out + \
            icache_hist_linear_out + dcache_hist_linear_out + page_hist_linear_out + \
            branch_linear_out + flag_linear_out
        self.inst_linear = nn.Linear(concat_dim, instruction_repr_dim)

        self.norm = nn.LayerNorm(instruction_repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type_feat = x[..., 0].long()
        reg_feat = x[..., 1:65].float()
        icache_hist_feat = x[..., 65:129].float()
        dcache_hist_feat = x[..., 129:193].float()
        page_hist_feat = x[..., 193:257].float()
        branch_feat = x[..., 257:289].float()
        flag_feat = x[..., 289:292].float()

        type_embed = self.type_embedding(type_feat)
        reg_repr = self.reg_linear(reg_feat)
        icache_hist_repr = self.icache_hist_linear(icache_hist_feat)
        dcache_hist_repr = self.dcache_hist_linear(dcache_hist_feat)
        page_hist_repr = self.page_hist_linear(page_hist_feat)
        branch_repr = self.branch_linear(branch_feat)
        flag_repr = self.flag_linear(flag_feat)
        concat = torch.cat(
            (type_embed, reg_repr, icache_hist_repr, dcache_hist_repr,
             page_hist_repr, branch_repr, flag_repr), dim=-1)
        concat = F.silu(concat)
        inst_repr = self.inst_linear(concat)
        inst_repr = self.norm(inst_repr)
        return F.silu(inst_repr)


class MultiTaskOutputHead(nn.Module):
    """
    Output heads for all tasks:
    1. fetch_cycle_class: 11 classes (class 0-9 for cycles 1-10, class 10 for 10+)
    2. fetch_cycle_regression: regression (used when cycle >= 11)
    3. exec_cycle_class: 21 classes (class 0-19 for cycles 1-20, class 20 for 20+)
    4. exec_cycle_regression: regression (used when cycle >= 21)
    5. branch_predict: 3 classes (softmax: correct/dir_wrong/target_wrong)
    6. icache_hit: 3 classes (softmax: L1/L2/Memory)
    7. dcache_hit: 3 classes (softmax: L1/L2/Memory)

    Output dim = 11 + 1 + 21 + 1 + 3 + 3 + 3 = 43
    """

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.out_linear1 = nn.Linear(input_dim, 256)
        self.out_linear2 = nn.Linear(256, 43)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.silu(self.norm(x))
        out = self.out_linear1(x)
        out = F.silu(out)
        out = self.out_linear2(out)

        # fetch_cycle prediction: classification + regression
        fetch_cycle_class_logits = out[..., 0:11]
        fetch_cycle_regression = F.softplus(out[..., 11])

        # exec_cycle prediction: classification + regression
        exec_cycle_class_logits = out[..., 12:33]
        exec_cycle_regression = F.softplus(out[..., 33])

        # Other tasks
        branch_mispred_logits = out[..., 34:37]
        icache_hit_logits = out[..., 37:40]
        dcache_hit_logits = out[..., 40:43]

        branch_mispred = branch_mispred_logits.argmax(dim=-1)
        icache_hit = F.softmax(icache_hit_logits, dim=-1)
        dcache_hit = F.softmax(dcache_hit_logits, dim=-1)

        # Compute final fetch_cycle prediction
        fetch_cycle_class_pred = fetch_cycle_class_logits.argmax(dim=-1)
        fetch_cycle = torch.where(
            fetch_cycle_class_pred < 10,
            (fetch_cycle_class_pred + 1).float(),
            fetch_cycle_regression * 100
        )

        # Compute final exec_cycle prediction
        exec_cycle_class_pred = exec_cycle_class_logits.argmax(dim=-1)
        exec_cycle = torch.where(
            exec_cycle_class_pred < 20,
            (exec_cycle_class_pred + 1).float(),
            exec_cycle_regression * 100
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
        QKV = rearrange(QKV, "... l (h d_k c) -> c ... h l d_k",
                        h=self.num_heads, c=3)
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
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_encoder: RotaryEmbedding, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, positional_encoder)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = x + self.dropout(self.attn(self.norm1(x)))
        ffn_out = attn_out + self.dropout(self.ffn(self.norm2(attn_out)))
        return ffn_out


class InstructionNet(nn.Module):
    def __init__(self, hidden_dim, dropout: float = 0.2):
        super().__init__()
        self.inst_encoder = InstructionEncoder(hidden_dim)
        self.RoPE = RotaryEmbedding(hidden_dim // 4)
        self.layers = nn.Sequential(
            *[TransformerBlock(hidden_dim, 4, hidden_dim * 8 // 3, self.RoPE, dropout)
              for _ in range(3)]
        )
        self.output_head = MultiTaskOutputHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.inst_encoder(x)
        x = self.layers(x)
        return self.output_head(x)
