import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)

        # Linear projections
        Q = self.w_q(x)  # (batch, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (batch, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # (batch, n_heads, seq_len, seq_len)

        # Apply causal mask (lower triangular) - this is the main mask for decoder-only
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, -1e9)

        # Apply padding mask if provided (mask out PAD tokens)
        if mask is not None:
            # mask shape: (batch_size, seq_len) -> need (batch_size, 1, 1, seq_len)
            # This masks out attention TO padding tokens
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(padding_mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch, n_heads, seq_len, d_k)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.w_o(context)

        return output


class FeedForward(nn.Module):
    """Position-wise Feed Forward Network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm architecture (like GPT)
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attention_out = self.attention(norm_x, mask)
        x = x + self.dropout(attention_out)

        # Feed forward with residual connection
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class VietnameseTransformer(nn.Module):
    """Decoder-only Transformer for Vietnamese Text Generation"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between embedding and output projection (optional but common)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Process through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)

        # Final layer norm
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        loss = None
        if return_loss and labels is not None:
            # Compute language modeling loss
            # Shift labels: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        """Generate text using the trained model"""

        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        generated_tokens = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Truncate sequence if it gets too long (prevent memory issues)
                if generated_tokens.size(1) > 512:
                    generated_tokens = generated_tokens[:, -256:]

                # Forward pass
                logits, _ = self.forward(generated_tokens, return_loss=False)

                # Get logits for last position
                next_token_logits = logits[:, -1, :].clone()

                # Apply temperature (avoid division by zero)
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / max(temperature, 1e-8)

                # Clamp logits to prevent extreme values
                next_token_logits = torch.clamp(next_token_logits, min=-50, max=50)

                # Apply top-k filtering
                if top_k > 0 and top_k < next_token_logits.size(-1):
                    top_k_actual = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, top_k_actual
                    )
                    # Create mask for top-k tokens
                    top_k_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    top_k_mask.scatter_(-1, top_k_indices, True)
                    # Set non-top-k tokens to very low value (not -inf to avoid numerical issues)
                    next_token_logits = next_token_logits.masked_fill(
                        ~top_k_mask, -50.0
                    )

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    # Use stable softmax
                    sorted_logits = (
                        sorted_logits - sorted_logits.max(dim=-1, keepdim=True)[0]
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Convert back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, -50.0
                    )

                # Final clamp before softmax
                next_token_logits = torch.clamp(next_token_logits, min=-50, max=50)

                # Sample or greedy decode
                if do_sample:
                    # Use stable softmax
                    next_token_logits = (
                        next_token_logits
                        - next_token_logits.max(dim=-1, keepdim=True)[0]
                    )
                    probs = F.softmax(next_token_logits, dim=-1)

                    # Check for valid probabilities
                    if (
                        torch.isnan(probs).any()
                        or torch.isinf(probs).any()
                        or (probs < 0).any()
                    ):
                        print(
                            f"Warning: Invalid probabilities at step {step}, using greedy decoding"
                        )
                        next_tokens = torch.argmax(
                            next_token_logits, dim=-1, keepdim=True
                        )
                    else:
                        # Add small epsilon to prevent all-zero probabilities
                        probs = probs + 1e-8
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        try:
                            next_tokens = torch.multinomial(probs, num_samples=1)
                        except RuntimeError as e:
                            print(
                                f"Warning: Multinomial sampling failed at step {step}: {e}"
                            )
                            print(f"Using greedy decoding instead")
                            next_tokens = torch.argmax(
                                next_token_logits, dim=-1, keepdim=True
                            )
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

                # Stop if EOS token is generated (check all samples in batch)
                if (next_tokens == eos_token_id).any():
                    break

        return generated_tokens


# Example usage
if __name__ == "__main__":
    # This would be used with your data preparation pipeline
    print("Vietnamese Decoder-Only Transformer")
    print("=====================================")

    # Example model configuration
    vocab_size = 5000
    model = VietnameseTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=128,
        dropout=0.1,
    )

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Example forward pass
    batch_size = 2
    seq_len = 10

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(dummy_input, labels=dummy_labels, return_loss=True)
        print(f"Output shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")

    # Example generation
    print("\n=== Testing Generation ===")
    generated = model.generate(
        torch.randint(0, vocab_size, (1, 5)),
        max_new_tokens=10,
        temperature=1.0,
        do_sample=True,
    )
    print(f"Generated shape: {generated.shape}")
