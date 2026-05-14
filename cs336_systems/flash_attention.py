import torch
import triton
import triton.language as tl


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 forward pass implemented with pure PyTorch tiled operations.
    Tile sizes are at least 16×16; inputs are assumed to be powers of 2 >= 16.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
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


class FlashAttentionTriton(torch.autograd.Function):
    """
    FlashAttention-2 forward pass implemented with pure PyTorch tiled operations.
    Tile sizes are at least 16×16; inputs are assumed to be powers of 2 >= 16.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False):
        BQ, N_QUERIES, D = Q.shape
        BK, N_KEYS, D = K.shape

        O = torch.zeros_like(Q)
        L = torch.zeros((BQ, N_QUERIES), dtype=torch.float32, device=Q.device)
        grid = (triton.cdiv(N_QUERIES, 32), BQ)
        flash_fwd_kernel[grid](
            Q_ptr=Q,
            K_ptr=K,
            V_ptr=V,
            O_ptr=O,
            L_ptr=L,
            stride_qb=Q.stride(0),
            stride_qq=Q.stride(1),
            stride_qd=Q.stride(2),
            stride_kb=K.stride(0),
            stride_kk=K.stride(1),
            stride_kd=K.stride(2),
            stride_vb=V.stride(0),
            stride_vk=V.stride(1),
            stride_vd=V.stride(2),
            stride_ob=O.stride(0),
            stride_oq=O.stride(1),
            stride_od=O.stride(2),
            stride_lb=L.stride(0),
            stride_lq=L.stride(1),
            N_QUERIES=N_QUERIES,
            N_KEYS=N_KEYS,
            scale=D**-0.5,
            D=D,
            Q_TILE_SIZE=32,
            K_TILE_SIZE=32,
            is_casual=is_casual,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_casual: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    # Q block pointer: points to the query_tile_index-th tile of queries
    # [B, Bq, D]
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # O block pointer: points to the query_tile_index-th tile of output
    # [B, Bq, D]
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L block pointer: 1D tensor, points to query_tile_index-th tile of logsumexp
    # [B, Bq]
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # K block pointer: starts at beginning, will advance in loop over key tiles
    # [B, Bk, D]
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # Start at first key tile
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V block pointer: starts at beginning, will advance in loop over key tiles
    # [B, Bk, D]
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),  # Start at first value tile
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # S's shape [B, Bq, Bk]
        S_block = tl.dot(q_block, tl.trans(K_block)) * scale

        if is_casual:
            row_offset = query_tile_index * Q_TILE_SIZE
            col_offset = j * K_TILE_SIZE
            rows = tl.arange(0, Q_TILE_SIZE) + row_offset
            cols = tl.arange(0, K_TILE_SIZE) + col_offset
            casual_mask = rows[:, None] >= cols[None, :]
            S_block = tl.where(casual_mask, S_block, 1e-6)

        # m's shape [B, Bq]
        m_j_1 = m_i
        m_j = tl.maximum(m_i, tl.max(S_block, axis=-1))
        m_i = m_j

        # P's shape [B, Bq, Bk]
        P_j = tl.exp(S_block - m_j[:, None])

        # l's shape [B, Bq]
        l_j_1 = l_i
        l_j = tl.exp(m_j_1 - m_j) * l_j_1 + tl.sum(P_j, axis=-1)
        l_i = l_j

        # O's shape [B, Bq, D]
        O_j_1 = O_i
        O_i = tl.exp(m_j_1 - m_j)[:, None] * O_j_1 + tl.dot(P_j, V_block)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    O_i = O_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))
