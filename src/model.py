"""
Decoder-Only Transformer (GPT-style) for Causal Language Modelling.
Shared across all curriculum phases.

Architecture — explicitly decoder-only:
  Each layer has ONLY:
    1. Masked self-attention  (causal, looks only at past tokens)
    2. Feed-forward network   (two linear layers with GELU)
  There is NO cross-attention, NO encoder, NO memory input.
  This is identical to GPT-2/GPT-3 architecture.

Why we do NOT use nn.TransformerDecoderLayer:
  PyTorch's TransformerDecoderLayer has 3 sub-layers:
    self-attn → cross-attn → FFN
  Cross-attention is for encoder-decoder models (T5, BART).
  In a decoder-only LM the cross-attention sub-layer is wasted.
  We implement our own GPTBlock so the architecture is explicit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    Tokens can only attend to themselves and earlier positions — never the future.
    No cross-attention, no encoder memory. Pure decoder-only.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model

        # Single fused projection for Q, K, V (more efficient than 3 separate)
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                           # (batch, seq_len, d_model)

        # Compute Q, K, V in one matmul, then split
        qkv = self.qkv_proj(x)                      # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=2)   # each (B, T, d_model)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention with causal mask (PyTorch 2.0 flash attn)
        # is_causal=True automatically builds the upper-triangular mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,                          # ← enforces decoder-only behaviour
        )                                            # (B, n_heads, T, head_dim)

        # Merge heads back: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear (same as GPT-2)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GPTBlock(nn.Module):
    """
    One decoder-only Transformer block (GPT-style):

        x → LayerNorm → CausalSelfAttention ──┐
        └──────────────────────────────────── + → x
        x → LayerNorm → FeedForward ──────── ─┐
        └──────────────────────────────────── + → x

    Pre-LayerNorm (norm_first) for better training stability at scale.
    NO cross-attention anywhere — purely decoder-only.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # self-attention residual
        x = x + self.ffn(self.ln2(x))    # feed-forward residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class DecoderOnlyTransformer(nn.Module):
    """
    GPT-style Decoder-Only Transformer for causal language modelling.

    Stack of GPTBlocks (self-attention + FFN only, no cross-attention).
    Uses learned positional embeddings and tied token/lm_head weights.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pad_id   = pad_id
        self.max_len  = max_len

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)   # learned positions (GPT-2 style)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Decoder stack (N × GPTBlock, each = self-attn + FFN) ─────────────
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output head ───────────────────────────────────────────────────────
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and lm_head weights (saves ~50M params)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        Args:
            x       : (B, T) token ids
            targets : (B, T) gold ids — if given, also returns cross-entropy loss

        Returns:
            logits : (B, T, vocab_size)
            loss   : scalar or None
        """
        B, T = x.shape
        assert T <= self.max_len, \
            f"Input length {T} exceeds model max_len {self.max_len}"

        # Token embeddings scaled by sqrt(d_model) + learned position embeddings
        pos = torch.arange(T, device=x.device).unsqueeze(0)          # (1, T)
        x   = self.emb_drop(
                  self.token_emb(x) * math.sqrt(self.d_model)
                  + self.pos_emb(pos)
              )                                                         # (B, T, d_model)

        # Pass through N decoder blocks (each: self-attn + FFN, causal)
        for block in self.blocks:
            x = block(x)

        x      = self.ln_final(x)
        logits = self.lm_head(x)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int = 200,
        min_new: int = 0,           # minimum tokens to generate before EOS is allowed
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.3,
        eos_id: int = None,
        max_seq_len: int = 512,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with:
          - Temperature scaling
          - Top-k filtering
          - Nucleus (top-p) sampling
          - Repetition penalty (prevents loops like 'Sarah Sarah Sarah...')
          - min_new: suppress EOS for the first N generated tokens
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new):
            ctx    = ids[:, -max_seq_len:]              # truncate to context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()           # (1, vocab_size) — last token only

            # ── Repetition penalty ────────────────────────────────────────────
            if repetition_penalty != 1.0:
                for token_id in set(ctx[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # ── Temperature ───────────────────────────────────────────────────
            logits = logits / max(temperature, 1e-8)

            # ── Top-k ─────────────────────────────────────────────────────────
            if top_k and top_k > 0:
                kth_val = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits  = logits.masked_fill(logits < kth_val, float("-inf"))

            # ── Top-p (nucleus) ───────────────────────────────────────────────
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_idx = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_idx] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            nxt   = torch.multinomial(probs, 1)
            ids   = torch.cat([ids, nxt], dim=1)

            # Only allow EOS after min_new tokens have been generated
            generated_so_far = ids.shape[1] - prompt_ids.shape[1]
            if eos_id is not None and nxt.item() == eos_id and generated_so_far >= min_new:
                break
            elif eos_id is not None and nxt.item() == eos_id and generated_so_far < min_new:
                # Replace the EOS with the second-best token
                probs[0, eos_id] = 0.0
                probs = probs / probs.sum()    # re-normalize
                nxt   = torch.multinomial(probs, 1)
                ids[:, -1] = nxt  # overwrite the EOS we just appended

        return ids


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    total = count_parameters(model)
    print(f"\n  Model Architecture: Decoder-Only Transformer (GPT-style)")
    print(f"  {'─'*46}")
    print(f"  Total trainable parameters : {total / 1e6:.2f} M")
    for name, mod in model.named_children():
        p = sum(x.numel() for x in mod.parameters() if x.requires_grad)
        if p > 0:
            print(f"    {name:<20s}  {p / 1e6:7.2f} M")
    print(f"  {'─'*46}")
    print(f"  Note: lm_head weights are TIED to token_emb (not double-counted)")
    print()
