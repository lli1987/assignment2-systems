import torch


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 forward pass implemented with pure PyTorch tiled operations.
    Tile sizes are at least 16×16; inputs are assumed to be powers of 2 >= 16.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q, K, V: (batch, seq_len, d_head)
        B, N, d = Q.shape
        scale = d ** -0.5

        Br = 32  # row tile size (queries)
        Bc = 32  # column tile size (keys/values)

        Tr = N // Br  # number of row tiles
        Tc = N // Bc  # number of column tiles

        O = torch.zeros_like(Q)
        L = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        for i in range(Tr):
            # Load i-th tile of Q: shape (B, Br, d)
            # Initialize running statistics for this tile: m_i, l_i, O_i
            pass

            for j in range(Tc):
                # Load j-th tile of K, V: shape (B, Bc, d)
                # Compute S_ij = Q_i @ K_j^T * scale: shape (B, Br, Bc)
                # Apply causal mask if is_causal (mask future positions)
                # Compute tile max m_ij and update running max m_i
                # Compute exp(S_ij - m_ij) and rescale previous O_i and l_i
                # Accumulate O_i += exp(S_ij - m_ij) @ V_j
                # Accumulate l_i (running sum of exponentials)
                pass

            # Normalise: O_i /= l_i
            # Store L_i = m_i + log(l_i) into L[i*Br:(i+1)*Br]
            pass

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
