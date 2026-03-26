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
    const_1_ptr,
    const_1_stride0: tl.constexpr,
    const_1_stride1: tl.constexpr,
    const_1_stride2: tl.constexpr,
    const_2_ptr,
    const_2_stride0: tl.constexpr,
    const_2_stride1: tl.constexpr,
    const_2_stride2: tl.constexpr,
    lv34_ptr,
    lv34_stride0: tl.constexpr,
    lv34_stride1: tl.constexpr,
    lv34_stride2: tl.constexpr,
    lv42_ptr,
    lv42_stride0: tl.constexpr,
    lv42_stride1: tl.constexpr,
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
    lv1 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv3 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv33 = tl.zeros((1, 16, BLOCK_M), dtype=tl.float16)
    lv35 = tl.zeros((1, 16), dtype=tl.float32)
    lv40 = tl.zeros((1, 16, 128), dtype=tl.float32)
    lv5 = tl.zeros((16, BLOCK_J), dtype=tl.float16)

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
    temp_7 = tl.expand_dims(lv1, 1)
    lv9 = tl.permute(temp_7, (1, 0, 2))
    temp_8 = tl.expand_dims(lv3, 1)
    lv10 = tl.permute(temp_8, (1, 0, 2))
    temp_9 = tl.expand_dims(lv5, 1)
    lv11 = tl.permute(temp_9, (1, 0, 2))
    offset_4 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (512 + tl.arange(0, 16))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
    tl.store(const_1_ptr + offset_4, lv10)
    offset_5 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (512 + tl.arange(0, 16))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
    tl.store(const_2_ptr + offset_5, lv11)
    # Sequential loop m from 0 to 528 with tile size BLOCK_M
    for m in range(0, 528, BLOCK_M):
        offset_6 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_4 = (m_indices < 528)[None, :, None]
        temp_10 = tl.load(const_1_ptr + offset_6, mask=mask_4, other=0.0)
        temp_11 = tl.permute(temp_10, (0, 2, 1))
        lv33 = (tl.dot(lv9, temp_11).to(tl.float16) + lv33).to(tl.float16)
        offset_7 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv34_stride0 + (tl.arange(0, 16))[None, :, None] * lv34_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv34_stride2
        mask_5 = (m_indices < 528)[None, None, :]
        tl.store(lv34_ptr + offset_7, tl.exp(lv33.to(tl.float32)), mask=mask_5)
        lv35 = (lv35 + tl.sum(tl.exp(lv33.to(tl.float32)), axis=2, dtype=tl.float32))
    lv35 = lv35 + 0.0
    # Sequential loop m from 0 to 528 with tile size BLOCK_M
    for m in range(0, 528, BLOCK_M):
        offset_8 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv34_stride0 + (tl.arange(0, 16))[None, :, None] * lv34_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv34_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_6 = (m_indices < 528)[None, None, :]
        temp_12 = tl.load(lv34_ptr + offset_8, mask=mask_6, other=0.0)
        temp_13 = tl.expand_dims(lv35, 2)
        temp_14 = tl.reshape(temp_13, (1, 16))
        offset_9 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
        mask_7 = (m_indices < 528)[None, :, None]
        temp_15 = tl.load(const_2_ptr + offset_9, mask=mask_7, other=0.0)
        lv40 = (lv40 + tl.dot((temp_12 / temp_14[:, :, None]), temp_15.to(tl.float32)))
    temp_16 = tl.permute(lv40, (1, 0, 2))
    offset_10 = (tl.arange(0, 16))[:, None] * lv42_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv42_stride1
    j_indices = j + tl.arange(0, BLOCK_J)
    mask_8 = (j_indices < 4096)[None, :]
    tl.store(lv42_ptr + offset_10, tl.reshape(temp_16, (16, 128)).to(tl.float16), mask=mask_8)


# Metadata for benchmark.py
TENSOR_PARAMS = ['const_1', 'const_2', 'lv34', 'lv42', 'p_k_proj_weight', 'p_q_proj_weight', 'p_v_proj_weight', 'x']
FP32_TENSOR_PARAMS = ['lv34']
BLOCK_PARAMS = ['block_i', 'block_m']

def forward(const_1, const_2, lv34, lv42, p_k_proj_weight, p_q_proj_weight, p_v_proj_weight, x, block_i=16, block_m=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4096 - 0 + 128 - 1) // 128,)](
        const_1,
        const_1.stride(0),
        const_1.stride(1),
        const_1.stride(2),
        const_2,
        const_2.stride(0),
        const_2.stride(1),
        const_2.stride(2),
        lv34,
        lv34.stride(0),
        lv34.stride(1),
        lv34.stride(2),
        lv42,
        lv42.stride(0),
        lv42.stride(1),
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
