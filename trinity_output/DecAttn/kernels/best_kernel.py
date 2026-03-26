import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_I': 32, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_I': 32, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_I': 32, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_I': 64, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_I': 64, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_I': 64, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_I': 128, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_I': 128, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_I': 128, 'BLOCK_M': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    T_softmax_exp_ptr,
    T_softmax_exp_stride0: tl.constexpr,
    T_softmax_exp_stride1: tl.constexpr,
    T_softmax_exp_stride2: tl.constexpr,
    const_1_ptr,
    const_1_stride0: tl.constexpr,
    const_1_stride1: tl.constexpr,
    const_1_stride2: tl.constexpr,
    const_2_ptr,
    const_2_stride0: tl.constexpr,
    const_2_stride1: tl.constexpr,
    const_2_stride2: tl.constexpr,
    lv33_ptr,
    lv33_stride0: tl.constexpr,
    lv33_stride1: tl.constexpr,
    lv33_stride2: tl.constexpr,
    lv39_ptr,
    lv39_stride0: tl.constexpr,
    lv39_stride1: tl.constexpr,
    p_k_proj_weight_ptr,
    p_k_proj_weight_stride0: tl.constexpr,
    p_k_proj_weight_stride1: tl.constexpr,
    p_q_proj_weight_ptr,
    p_q_proj_weight_stride0: tl.constexpr,
    p_q_proj_weight_stride1: tl.constexpr,
    p_v_proj_weight_ptr,
    p_v_proj_weight_stride0: tl.constexpr,
    p_v_proj_weight_stride1: tl.constexpr,
    x_ptr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Allocate intermediate tensors
    T_softmax_expsum = tl.zeros((1, 16), dtype=tl.float32)
    T_softmax_maxelem = tl.zeros((1, 16), dtype=tl.float16)
    lv1 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv3 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv37 = tl.zeros((1, 16, 128), dtype=tl.float32)
    lv5 = tl.zeros((16, BLOCK_J), dtype=tl.float16)

    # Initialize kernel accumulators
    # Parallel loop j from 0 to lv1_dim1 with tile size BLOCK_J
    # Executed across grid dimension 0
    j = 0 + tl.program_id(0) * BLOCK_J
    
    # Sequential loop i from 0 to 4096 with tile size BLOCK_I
    for i in range(0, 4096, BLOCK_I):
        offset_0 = (tl.arange(0, 16))[:, None] * x_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * x_stride1
        i_indices = i + tl.arange(0, BLOCK_I)
        mask_0 = (i_indices < 4096)[None, :]
        temp_0 = tl.load(x_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (j + tl.arange(0, BLOCK_J))[:, None] * p_q_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_q_proj_weight_stride1
        j_indices = j + tl.arange(0, BLOCK_J)
        mask_1 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_1 = tl.load(p_q_proj_weight_ptr + offset_1, mask=mask_1, other=0.0)
        temp_2 = tl.trans(temp_1)
        lv1 = (lv1 + tl.dot(temp_0, temp_2).to(tl.float16)).to(tl.float16)
        offset_2 = (j + tl.arange(0, BLOCK_J))[:, None] * p_k_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_k_proj_weight_stride1
        mask_2 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_3 = tl.load(p_k_proj_weight_ptr + offset_2, mask=mask_2, other=0.0)
        temp_4 = tl.trans(temp_3)
        lv3 = (lv3 + tl.dot(temp_0, temp_4).to(tl.float16)).to(tl.float16)
        offset_3 = (j + tl.arange(0, BLOCK_J))[:, None] * p_v_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_v_proj_weight_stride1
        mask_3 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_5 = tl.load(p_v_proj_weight_ptr + offset_3, mask=mask_3, other=0.0)
        temp_6 = tl.trans(temp_5)
        lv5 = (lv5 + tl.dot(temp_0, temp_6).to(tl.float16)).to(tl.float16)

    lv6 = tl.expand_dims(lv1, 1)

    lv7 = tl.expand_dims(lv3, 1)

    lv8 = tl.expand_dims(lv5, 1)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_4 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
        tl.store(const_1_ptr + offset_4, tl.permute(lv7, (1, 0, 2)))
        offset_5 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
        tl.store(const_2_ptr + offset_5, tl.permute(lv8, (1, 0, 2)))
        temp_7 = tl.permute(lv6, (1, 0, 2))
        offset_6 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_4 = (m_indices < 1024)[None, :, None]
        temp_8 = tl.load(const_1_ptr + offset_6, mask=mask_4, other=0.0)
        temp_9 = tl.permute(temp_8, (0, 2, 1))
        offset_7 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv33_stride0 + (tl.arange(0, 16))[None, :, None] * lv33_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv33_stride2
        mask_5 = (m_indices < 1024)[None, None, :]
        temp_10 = tl.load(lv33_ptr + offset_7, mask=mask_5, other=0.0)
        offset_8 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv33_stride0 + (tl.arange(0, 16))[None, :, None] * lv33_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv33_stride2
        tl.store(lv33_ptr + offset_8, (tl.dot(temp_7, temp_9).to(tl.float16) + temp_10).to(tl.float16), mask=mask_5)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_9 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv33_stride0 + (tl.arange(0, 16))[None, :, None] * lv33_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv33_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_6 = (m_indices < 1024)[None, None, :]
        temp_11 = tl.load(lv33_ptr + offset_9, mask=mask_6, other=0.0)
        T_softmax_maxelem = tl.maximum(T_softmax_maxelem, tl.max(temp_11, axis=2)).to(tl.float16)
    T_softmax_maxelem = T_softmax_maxelem + 0.0
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_10 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv33_stride0 + (tl.arange(0, 16))[None, :, None] * lv33_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv33_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_7 = (m_indices < 1024)[None, None, :]
        temp_12 = tl.load(lv33_ptr + offset_10, mask=mask_7, other=0.0)
        offset_11 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * T_softmax_exp_stride0 + (tl.arange(0, 16))[None, :, None] * T_softmax_exp_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * T_softmax_exp_stride2
        mask_8 = (m_indices < 1024)[None, None, :]
        tl.store(T_softmax_exp_ptr + offset_11, tl.exp((temp_12 - T_softmax_maxelem[:, :, None]).to(tl.float32)), mask=mask_8)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_12 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * T_softmax_exp_stride0 + (tl.arange(0, 16))[None, :, None] * T_softmax_exp_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * T_softmax_exp_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_9 = (m_indices < 1024)[None, None, :]
        temp_13 = tl.load(T_softmax_exp_ptr + offset_12, mask=mask_9, other=0.0)
        T_softmax_expsum = (tl.sum(temp_13, axis=2, dtype=tl.float32) + T_softmax_expsum)
    T_softmax_expsum = T_softmax_expsum + 0.0
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_13 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * T_softmax_exp_stride0 + (tl.arange(0, 16))[None, :, None] * T_softmax_exp_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * T_softmax_exp_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_10 = (m_indices < 1024)[None, None, :]
        temp_14 = tl.load(T_softmax_exp_ptr + offset_13, mask=mask_10, other=0.0)
        offset_14 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
        mask_11 = (m_indices < 1024)[None, :, None]
        temp_15 = tl.load(const_2_ptr + offset_14, mask=mask_11, other=0.0)
        lv37 = (lv37 + tl.dot((temp_14 / T_softmax_expsum[:, :, None]), temp_15.to(tl.float32)))
    temp_16 = tl.permute(lv37, (1, 0, 2))
    offset_15 = (tl.arange(0, 16))[:, None] * lv39_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv39_stride1
    j_indices = j + tl.arange(0, BLOCK_J)
    mask_12 = (j_indices < 4096)[None, :]
    tl.store(lv39_ptr + offset_15, tl.reshape(temp_16, (16, 128)).to(tl.float16), mask=mask_12)


# Metadata for benchmark.py
TENSOR_PARAMS = ['T_softmax_exp', 'const_1', 'const_2', 'lv33', 'lv39', 'p_k_proj_weight', 'p_q_proj_weight', 'p_v_proj_weight', 'x']
FP32_TENSOR_PARAMS = ['T_softmax_exp']
BLOCK_PARAMS = ['block_i', 'block_m']

def forward(T_softmax_exp, const_1, const_2, lv33, lv39, p_k_proj_weight, p_q_proj_weight, p_v_proj_weight, x, block_i=16, block_m=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4096 - 0 + 128 - 1) // 128,)](
        T_softmax_exp,
        T_softmax_exp.stride(0),
        T_softmax_exp.stride(1),
        T_softmax_exp.stride(2),
        const_1,
        const_1.stride(0),
        const_1.stride(1),
        const_1.stride(2),
        const_2,
        const_2.stride(0),
        const_2.stride(1),
        const_2.stride(2),
        lv33,
        lv33.stride(0),
        lv33.stride(1),
        lv33.stride(2),
        lv39,
        lv39.stride(0),
        lv39.stride(1),
        p_k_proj_weight,
        p_k_proj_weight.stride(0),
        p_k_proj_weight.stride(1),
        p_q_proj_weight,
        p_q_proj_weight.stride(0),
        p_q_proj_weight.stride(1),
        p_v_proj_weight,
        p_v_proj_weight.stride(0),
        p_v_proj_weight.stride(1),
        x,
        x.stride(0),
        x.stride(1),
        # BLOCK_I, BLOCK_M are provided by autotune,
        BLOCK_J=128,
        # BLOCK_I is automatically set by autotune,
        # BLOCK_M is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
