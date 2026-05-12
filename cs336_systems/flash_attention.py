import torch
import torch.nn.functional as fn


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 forward pass implemented with pure PyTorch tiled operations.
    Tile sizes are at least 16×16; inputs are assumed to be powers of 2 >= 16.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q, K, V: (batch, seq_len, d_head)
        B, N, d = Q.shape
        scale = d**-0.5

        Br = 32  # row tile size (queries)
        Bc = 32  # column tile size (keys/values)

        Tr = N // Br  # number of row tiles
        Tc = N // Bc  # number of column tiles

        O = torch.zeros_like(Q)
        L = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        for i in range(Tr):
            # Load i-th tile of Q: shape (B, Br, d)
            Q_i = Q[:, i * Br : (i + 1) * Br, :]

            # Initialize running statistics for this tile: m_i, l_i, O_i
            O_i = torch.zeros(B, Br, d, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((B, Br), -float("inf"), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, Br, device=Q.device, dtype=Q.dtype)

            for j in range(Tc):
                # Load j-th tile of K, V: shape (B, Bc, d)
                K_j = K[:, j * Bc : (j + 1) * Bc, :]
                V_j = V[:, j * Bc : (j + 1) * Bc, :]

                # Compute S_ij = Q_i @ K_j^T * scale: shape (B, Br, Bc)
                S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale

                # Apply causal mask if is_causal (mask future positions)
                # if is_causal:
                #     # Create causal mask for this tile
                #     row_idx = i * Br
                #     col_idx = j * Bc
                #     causal_mask = (
                #         torch.arange(Br, device=Q.device).unsqueeze(1) + row_idx
                #         >= torch.arange(Bc, device=Q.device).unsqueeze(0) + col_idx
                #     )
                #     S_ij = S_ij.masked_fill(~causal_mask, -float("inf"))

                # Compute tile max m_ij and update running max m_i
                m_ij = torch.max(S_ij, dim=-1).values  # (B, Br)
                m_i_new = torch.max(m_i, m_ij)  # (B, Br)

                # Compute P_ij = exp(S_ij - m_i_new)
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # (B, Br, Bc)

                # Rescale previous O_i and l_i, then accumulate
                # Algorithm: O_i = diag(exp(m_i_old - m_i_new)) @ O_i_old + P_ij @ V_j
                # With batches, diag(v) @ M is equivalent to v.unsqueeze(-1) * M (broadcasting)
                alpha = torch.exp(m_i - m_i_new).unsqueeze(-1)  # (B, Br, 1)
                O_i = alpha * O_i + P_ij @ V_j

                # Update running sum l_i
                l_i = torch.exp(m_i - m_i_new) * l_i + torch.sum(P_ij, dim=-1)

                # Update running max
                m_i = m_i_new

            # Normalize: O_i = diag(l_i)^(-1) @ O_i
            # With batches: diag(1/l_i) @ O_i is equivalent to O_i / l_i.unsqueeze(-1)
            O_i = O_i / l_i.unsqueeze(-1)

            # Compute L_i = m_i + log(l_i)
            L_i = m_i + torch.log(l_i)

            # Write O_i to global memory as the i-th tile of O
            O[:, i * Br : (i + 1) * Br, :] = O_i
            # Write L_i to global memory as the i-th tile of L
            L[:, i * Br : (i + 1) * Br] = L_i

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
