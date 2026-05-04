import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class BranchPredictor(nn.Module):
    def __init__(self, hist_dim=32, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ICachePredictor(nn.Module):
    def __init__(self, hist_dim=64, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCachePredictor(nn.Module):
    def __init__(self, dcache_hist_dim=64, page_hist_dim=64, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dcache_hist_dim + page_hist_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, dcache_hist: torch.Tensor, page_hist: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([dcache_hist, page_hist], dim=-1))


def build_main_input(type_reg_flags, branch_onehot, icache_onehot, dcache_onehot):
    return torch.cat([type_reg_flags, branch_onehot, icache_onehot, dcache_onehot], dim=-1)


class InstructionEncoder(nn.Module):
    def __init__(self, instruction_repr_dim,
                 type_vocab_size=157,
                 type_embed_dim=256,
                 reg_linear_out=192,
                 flag_linear_out=32):
        super().__init__()

        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.reg_linear = nn.Linear(64, reg_linear_out)
        self.flag_linear = nn.Linear(3, flag_linear_out)

        # type(256) + reg(192) + branch_onehot(3) + icache_onehot(3) + dcache_onehot(3) + flags(32) = 489
        concat_dim = type_embed_dim + reg_linear_out + 3 + 3 + 3 + flag_linear_out
        self.inst_linear = nn.Linear(concat_dim, instruction_repr_dim)

        self.norm = nn.LayerNorm(instruction_repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x layout: type(1) + reg(64) + branch_oh(3) + icache_oh(3) + dcache_oh(3) + flags(3) = 77
        type_feat = x[..., 0].long()
        reg_feat = x[..., 1:65].float()
        branch_feat = x[..., 65:68].float()
        icache_feat = x[..., 68:71].float()
        dcache_feat = x[..., 71:74].float()
        flag_feat = x[..., 74:77].float()

        type_embed = self.type_embedding(type_feat)
        reg_repr = self.reg_linear(reg_feat)
        flag_repr = self.flag_linear(flag_feat)
        concat = torch.cat(
            (type_embed, reg_repr, branch_feat, icache_feat,
             dcache_feat, flag_repr), dim=-1)
        concat = F.silu(concat)
        inst_repr = self.inst_linear(concat)
        inst_repr = self.norm(inst_repr)
        return F.silu(inst_repr)


class WindowTotalOutputHead(nn.Module):
    """Predicts total fetch cycles for the effective window via pooling."""

    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, loss_start: int = 8) -> dict[str, torch.Tensor]:
        effective = x[:, loss_start:]  # (batch, eff_len, input_dim)
        eff_len = effective.shape[1]
        pooled = effective.mean(dim=1)  # (batch, input_dim)
        out = F.silu(self.norm(pooled))
        out = F.silu(self.fc1(out))
        avg = F.softplus(self.fc2(out)).squeeze(-1)  # (batch,) avg cycles per instruction
        return {"fetch_cycle_avg": avg, "eff_len": eff_len}


class MultiHeadSelfAttention(nn.Module):
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
    """Main model: predicts total fetch latency for a window of instructions."""
    def __init__(self, hidden_dim, dropout: float = 0.2):
        super().__init__()
        self.inst_encoder = InstructionEncoder(hidden_dim)
        self.RoPE = RotaryEmbedding(hidden_dim // 4)
        self.layers = nn.Sequential(
            *[TransformerBlock(hidden_dim, 4, hidden_dim * 8 // 3, self.RoPE, dropout)
              for _ in range(3)]
        )
        self.output_head = WindowTotalOutputHead(hidden_dim)

    def forward(self, x: torch.Tensor, loss_start: int = 8) -> dict[str, torch.Tensor]:
        x = self.inst_encoder(x)
        x = self.layers(x)
        return self.output_head(x, loss_start)
