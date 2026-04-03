import os
import math
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════
# common
# ══════════════════════════════════════════════════════════════════

# 定义 MXFP4 所有有效值（按编码顺序 0x0 ~ 0xF）
MXFP4_VALUES = torch.tensor([
    0.0,   # 0x0
    0.5,   # 0x1
    1.0,   # 0x2
    1.5,   # 0x3
    2.0,   # 0x4
    3.0,   # 0x5
    4.0,   # 0x6
    6.0,   # 0x7
    -0.0,  # 0x8  （注意：-0.0 == 0.0 在数值上，但可区分）
    -0.5,  # 0x9
    -1.0,  # 0xA
    -1.5,  # 0xB
    -2.0,  # 0xC
    -3.0,  # 0xD
    -4.0,  # 0xE
    -6.0,  # 0xF
], dtype=torch.float32)



def save_tensor_to_bin(tensor, filename, op_name):
    """
    保存 tensor 为 .bin 文件（原始字节）：
        - bf16 → uint16（按位 reinterpret）
        - fp8  → uint8（按位 reinterpret）
        - int8 → int8（直接保存）
    """
    SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if tensor.dtype == torch.float8_e4m3fn:
        # FP8 存储为 uint8（按位 reinterpret）
        data_uint = tensor.cpu().contiguous().view(torch.uint8)
        data_to_save = data_uint.numpy()
    elif tensor.dtype == torch.bfloat16:
        # BF16 存储为 uint16（按位 reinterpret）
        data_uint = tensor.cpu().contiguous().view(torch.uint16)
        data_to_save = data_uint.numpy()
    elif tensor.dtype == torch.int8:
        # int8 直接保存为 int8 字节
        data_to_save = tensor.cpu().contiguous().numpy().astype('int8')
    elif tensor.dtype == torch.uint8:
        # uint8 直接保存为 uint8 字节
        data_to_save = tensor.cpu().contiguous().numpy().astype('uint8')
    elif tensor.dtype == torch.uint16:
        # uint16 直接保存为 uint16 字节
        data_to_save = tensor.cpu().contiguous().numpy().astype('uint16')
    elif tensor.dtype == torch.float32:
        # float32 直接保存为 float32 字节
        data_to_save = tensor.cpu().contiguous().numpy().astype('float32')
    elif tensor.dtype == torch.int32:
        # int32 直接保存为 int32 字节
        data_to_save = tensor.cpu().contiguous().numpy().astype('int32')
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")

    data_to_save.tofile(filepath)
    print(f"Saved {tensor.shape} tensor to '{filepath}' (dtype: {data_to_save.dtype})")

#-----------待测试------------------ IGNORE ------------------
def process_aligned(tensor, dim=0, align_bytes=128, bit_width=8):
    """
    对张量的指定维度进行字节对齐填充，并将结果保存到二进制文件。

    Args:
        weight (torch.Tensor): 待处理的张量。
        filename (str): 输出文件名。
        op_name (str): 算子名称（透传给 save_tensor_to_bin）。
        dim (int): 需要对齐的维度索引。
        align_bytes (int): 对齐的字节数，支持 16, 32, 128, 256, 512。
        bit_width (int): 每个元素的位宽，支持 4, 8, 16（bfloat16 视为 16）。
    """
    # 计算每个元素的字节数
    elem_size = bit_width / 8.0          # 4bit -> 0.5, 8bit -> 1, 16bit -> 2
    L = tensor.shape[dim]                # 当前维度的长度（元素个数）
    cur_bytes = L * elem_size            # 当前维度占用的总字节数

    # 计算对齐后需要的总字节数
    aligned_bytes = math.ceil(cur_bytes / align_bytes) * align_bytes
    new_L = int(round(aligned_bytes / elem_size))   # 对齐后的元素个数（必然为整数）
    padding_size = new_L - L

    # 构建 torch.nn.functional.pad 所需的填充参数
    # pad 参数的顺序为： (左填充_最后一维, 右填充_最后一维, 左填充_倒数第二维, ..., 左填充_第一维, 右填充_第一维)
    pad_args = [0] * (2 * tensor.ndim)
    rev_idx = tensor.ndim - 1 - dim      # 从后往前数的维度索引
    pad_args[2 * rev_idx + 1] = padding_size   # 只对右侧进行填充，左侧为0

    # 执行填充
    padded_tensor = F.pad(tensor, tuple(pad_args))
    return padded_tensor
    


# ══════════════════════════════════════════════════════════════════
# activate
# ══════════════════════════════════════════════════════════════════
def act_slicehw_layout(act_tensor, filename, op_name, block1 = 16, block2 = 16,reshape = False):
    """
    将 scales_bf16 (256, 1024) ，然后按 16x16 元素（每块 512 字节）切分，
    按行优先顺序保存每个 block 到 bin 文件中。s
    """
    if reshape :
        if len(act_tensor.shape) == 3:
            B, M, K = act_tensor.shape
            act_tensor  = act_tensor.reshape(B*M, K)
            M = B*M
        elif len(act_tensor.shape) == 2:
            M, K = act_tensor.shape
        else:
            raise ValueError("act_tensor must be 2D or 3D tensor")

        # block = 16

        # -----------------------------
        # 1. 计算 padding 后的尺寸
        # -----------------------------
        pad_row = math.ceil(M / block1) * block1
        pad_col = math.ceil(K / block2) * block2

        # -----------------------------
        # 2. 创建 padding tensor
        # -----------------------------
        padded = torch.zeros(
            pad_row,
            pad_col,
            dtype=act_tensor.dtype,
            device=act_tensor.device
        )

        # -----------------------------
        # 3. 拷贝原始数据
        # -----------------------------
        padded[:M, :K] = act_tensor

        # -----------------------------
        # 4. 计算 block 数量
        # -----------------------------
        n_block_m = pad_row // block1
        n_block_k = pad_col // block2


        # 拆分为 16x16 blocks: (4, 128, 16, 16)
        scales_block_matrix = padded.view(n_block_m, block1, n_block_k, block2).permute(0, 2, 1, 3).contiguous()

        SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
        save_dir = os.path.join(SAVE_ROOT, op_name)
        os.makedirs(save_dir, exist_ok=True)
        scales_bin_path = os.path.join(save_dir, filename)

        with open(scales_bin_path, 'wb') as f:
            for j in range(n_block_k):      # 4
                for i in range(n_block_m):  # 128
                    block = scales_block_matrix[i, j]  # (16, 16)
                    data_uint16 = block.cpu().contiguous().view(torch.uint16)
                    f.write(data_uint16.numpy().tobytes())

        print(f"Saved blocked scales to '{scales_bin_path}', total size: {os.path.getsize(scales_bin_path)} bytes")
    else: 
        if len(act_tensor.shape) == 3:
            B, M, K = act_tensor.shape
        elif len(act_tensor.shape) == 2:
            act_tensor = act_tensor.unsqueeze(0)
            B, M, K = act_tensor.shape
        else:
            raise ValueError("act_tensor must be 2D or 3D tensor")

        # block = 16

        # -----------------------------
        # 1. 计算 padding 后的尺寸
        # -----------------------------
        pad_row = math.ceil(M / block1) * block1
        pad_col = math.ceil(K / block2) * block2

        # -----------------------------
        # 2. 创建 padding tensor
        # -----------------------------
        padded = F.pad(act_tensor, (0, pad_col - K, 0, pad_row - M, 0, 0))

        # -----------------------------
        # 4. 计算 block 数量
        # -----------------------------
        n_block_m = pad_row // block1
        n_block_k = pad_col // block2


        # 拆分为 16x16 blocks: (4, 128, 16, 16)
        scales_block_matrix = padded.view(B, n_block_m, block1, n_block_k, block2).permute(0, 1, 3, 2, 4).contiguous()

        SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
        save_dir = os.path.join(SAVE_ROOT, op_name)
        os.makedirs(save_dir, exist_ok=True)
        scales_bin_path = os.path.join(save_dir, filename)

        with open(scales_bin_path, 'wb') as f:
            for b in range(B):
                for j in range(n_block_k):      # 4
                    for i in range(n_block_m):  # 128
                        block = scales_block_matrix[b][i, j]  # (16, 16)
                        data_uint16 = block.cpu().contiguous().view(torch.uint16)
                        f.write(data_uint16.numpy().tobytes())

        print(f"Saved blocked scales to '{scales_bin_path}', total size: {os.path.getsize(scales_bin_path)} bytes")

# ══════════════════════════════════════════════════════════════════
# weight
# ══════════════════════════════════════════════════════════════════

def process_4bit_packed_int8(proj_weight, n_row, n_col):

    is_2d = proj_weight.dim() == 2

    # ------------------------------------------------
    # 1. 统一成 3D
    # ------------------------------------------------
    if is_2d:
        proj_weight = proj_weight.unsqueeze(0)

    B, _, _ = proj_weight.shape

    block_h = 32
    block_w = 16

    # -----------------------------------
    # 1. 计算 padding 后尺寸
    # -----------------------------------
    pad_row = math.ceil(n_row / block_h) * block_h
    pad_col = math.ceil(n_col / block_w) * block_w

    pad_h = pad_row - n_row
    pad_w = pad_col - n_col

    # -----------------------------------
    # 2. padding
    # -----------------------------------
    
    padded = F.pad(proj_weight, (0, pad_w, 0, pad_h))

    # ------------------------------------------------
    # 2. reshape 做 int4 pack
    # ------------------------------------------------
    proj_weight_reshaped = padded.reshape(B, n_row // 2, 2, n_col).to(torch.int8)

    low_part  = proj_weight_reshaped[:, :, 0, :]
    high_part = proj_weight_reshaped[:, :, 1, :] << 4

    packed = high_part | (low_part & 0x0F)
    packed_int8 = packed.to(torch.int8)

    # shape
    # (B, n_row/2, n_col)

    # ------------------------------------------------
    # 3. block split
    # ------------------------------------------------
    block_size = 32

    blocks = torch.split(
        packed_int8,
        split_size_or_sections=block_size,
        dim=1
    )

    # blocks: tuple
    # each shape (B, 32, n_col)

    # ------------------------------------------------
    # 4. transpose block
    # ------------------------------------------------
    transposed_blocks = [
        blk.transpose(1, 2) for blk in blocks
    ]

    # each -> (B, n_col, 32)

    # ------------------------------------------------
    # 5. stack blocks
    # ------------------------------------------------
    stacked_blocks = torch.stack(transposed_blocks, dim=1)

    # shape
    # (B, n_blocks, n_col, 32)

    # ------------------------------------------------
    # 6. flatten
    # ------------------------------------------------
    flat_blocked = stacked_blocks.reshape(-1)

    return flat_blocked


def map_bf16_fp4_to_4bit(A: torch.Tensor) -> torch.Tensor:
    """
    将已经是 MXFP4 值的 bf16 张量 A 直接映射为 4-bit 编码。
    
    要求：A 中每个元素必须严格等于 MXFP4_VALUES 中的某个值。
    
    Args:
        A: torch.Tensor, dtype=torch.bfloat16 or float32, any shape
    
    Returns:
        codes: torch.Tensor of same shape, dtype=torch.uint8, values in 0~15
    """
    device = A.device
    shape = A.shape

    # 转为 float32（bf16 在 CPU 上可能无法精确比较）
    A_f32 = A.to(torch.float32)

    # 处理 -0.0：在 float32 中，-0.0 和 0.0 数值相等，但我们需要区分
    # 方法：检查符号位
    is_neg_zero = (A_f32 == 0.0) & (torch.signbit(A_f32))
    
    # 将 -0.0 显式替换为一个唯一标识值（比如 -1e-9），避免与 +0.0 混淆
    # 但我们知道 MXFP4 中只有两个零：+0.0 (0x0) 和 -0.0 (0x8)
    # 所以先正常匹配，再修正 -0.0
    A_for_match = A_f32.clone()
    
    # 广播比较：(N, 1) == (1, 16) → (N, 16) bool
    flat_A = A_for_match.flatten()  # (N,)
    matches = (flat_A.unsqueeze(1) == MXFP4_VALUES.to(device).unsqueeze(0))  # (N, 16)

    # 每行应恰好有一个 True
    if not matches.any(dim=1).all():
        raise ValueError("Some elements in A are not valid MXFP4 values!")

    # 获取编码索引（argmax on bool works because only one True per row）
    codes_flat = matches.to(torch.uint8).argmax(dim=1)  # (N,)

    # 修正：如果原始值是 -0.0，则强制设为 0x8
    neg_zero_mask = is_neg_zero.flatten()
    if neg_zero_mask.any():
        codes_flat[neg_zero_mask] = 0x8

    return codes_flat.reshape(shape).to(torch.uint8)

def encode_weight_to_mxfp4(x):
    """
    将一个 float32 标量编码为最接近的 MXFP4 (E2M1) 4-bit 索引 (0~15)
    不使用 LUT，直接计算所有可能值并取最近。
    支持向量化输入（tensor）
    """
    # MXFP4 E2M1 非零值定义：
    # sign: ±1
    # exponent e ∈ {-1, 0, 1, 2}  (bias=1 → stored as 0~3)
    # mantissa: 1.0 or 1.5
    # total non-zero values: 2 (sign) * 4 (exp) * 2 (mantissa) = 16, but 0 is separate → 15 non-zero
    device = x.device
    dtype = torch.float32

    # 特殊处理 0
    zero_mask = (x == 0)

    # 所有可能的正数值（15 个非零值中的正数部分）
    exp_vals = torch.tensor([-1, 0, 1, 2], dtype=dtype, device=device)  # e
    mant_vals = torch.tensor([1.0, 1.5], dtype=dtype, device=device)     # 1 + f*0.5

    # 生成所有正候选值: [4, 2] → flatten to [8]
    pos_candidates = (2.0 ** exp_vals[:, None]) * mant_vals[None, :]  # [4,2]
    pos_candidates = pos_candidates.flatten()  # [8]

    # 添加负值 → [16] values: [pos..., neg...]
    all_candidates = torch.cat([pos_candidates, -pos_candidates], dim=0)  # [16]

    # 对应的 4-bit 索引（按标准 E2M1 编码顺序）
    # 我们需要构建从 candidate value → index 的映射
    # 手动构建索引列表（与 E2M1 编码一致）
    indices_list = []
    for s in [0, 1]:  # sign bit
        for e in range(4):  # stored exponent 0~3 → actual e = e - 1
            for m in [0, 1]:
                idx = (s << 3) | (e << 1) | m
                indices_list.append(idx)
    # indices_list 长度为 16，对应 all_candidates 顺序
    candidate_indices = torch.tensor(indices_list, dtype=torch.uint8, device=device)  # [16]

    # 扩展 x 为 [..., 1]，candidates 为 [16]
    x_exp = x.unsqueeze(-1).to(torch.float32)  # [..., 1]
    cand_exp = all_candidates.view(1, -1)  # [1, 16]

    # 计算绝对误差
    diff = torch.abs(x_exp - cand_exp)  # [..., 16]
    best_idx = torch.argmin(diff, dim=-1)  # [...]

    # 获取对应的 4-bit 索引
    result_indices = candidate_indices[best_idx]

    # 处理原始为 0 的情况 → 强制设为 0
    result_indices = torch.where(zero_mask, torch.tensor(0, dtype=torch.uint8, device=device), result_indices)

    return result_indices

#-----------待测试------------------ IGNORE ------------------
def process_bf16_weight(proj_weight, n_row, n_col):
    # ------------------------------------------------
    # 1. 计算 padding 后尺寸
    # ------------------------------------------------
    block_size = 16

    pad_row = math.ceil(n_row / block_size) * block_size
    pad_col = math.ceil(n_col / block_size) * block_size

    pad_h = pad_row - n_row
    pad_w = pad_col - n_col

    # ------------------------------------------------
    # 2. padding
    # ------------------------------------------------
    padded = F.pad(proj_weight, (0, pad_w, 0, pad_h))

    block_size = 16
    blocks = torch.split(proj_weight, split_size_or_sections=block_size, dim=0)  # tuple of (K // 16) x [16, N]
    transposed_blocks = [blk.transpose(0, 1) for blk in blocks]  # each [16, N] -> [N, 16]
    stacked_blocks = torch.stack(transposed_blocks, dim=0)  # shape: [K // 16, N, 16]
    flat_blocked = stacked_blocks.reshape(-1)  # shape: ((K // 16) * N * 16,)

    return flat_blocked

#-----------待测试------------------ IGNORE ------------------
def process_norm_weight(weight, filename, op_name):

    block_size = 128
    padding_size = math.ceil(weight.shape[0] / block_size) * block_size - weight.shape[0]
    pad = F.pad(weight, (0, padding_size))
    save_tensor_to_bin(pad, filename, op_name)


# ══════════════════════════════════════════════════════════════════
# scale
# ══════════════════════════════════════════════════════════════════

def save_int8_scales_blocked(scales_int8, n_row, n_col, filename, op_name):
    """
    支持:
    1) scales_int8 为 2D 或 3D tensor
    2) 自动 padding 到 (32,16) block 对齐

    block size:
        32 x 16
        每 block = 512 bytes (bf16)
    """

    block_h = 32
    block_w = 16

    # -----------------------------------
    # 1. 计算 padding 后尺寸
    # -----------------------------------
    pad_row = math.ceil(n_row / block_h) * block_h
    pad_col = math.ceil(n_col / block_w) * block_w

    pad_h = pad_row - n_row
    pad_w = pad_col - n_col

    # -----------------------------------
    # 2. padding
    # -----------------------------------
    if scales_int8.dim() == 2:

        padded = F.pad(scales_int8, (0, pad_w, 0, pad_h))
        padded = padded.unsqueeze(0)  # 统一成 3D

    elif scales_int8.dim() == 3:

        padded = F.pad(scales_int8, (0, pad_w, 0, pad_h))

    else:
        raise ValueError("scales_int8 must be 2D or 3D tensor")

    B, H, W = padded.shape

    # -----------------------------------
    # 3. block 数量
    # -----------------------------------
    n_block_h = H // block_h
    n_block_w = W // block_w

    # -----------------------------------
    # 4. reshape -> block matrix
    # (B, block_h, block_w, 32,16)
    # -----------------------------------
    scales_block_matrix = (
        padded.view(B, n_block_h, block_h, n_block_w, block_w)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    # shape
    # (B, n_block_h, n_block_w, 32,16)

    # -----------------------------------
    # 5. 保存路径
    # -----------------------------------
    SAVE_ROOT = "/root/autodl-tmp/project/golden_data"

    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)

    scales_bin_path = os.path.join(save_dir, filename)

    # -----------------------------------
    # 6. 写入 block
    # -----------------------------------
    with open(scales_bin_path, 'wb') as f:

        for b in range(B):
            for i in range(n_block_h):
                for j in range(n_block_w):

                    block = scales_block_matrix[b, i, j]  # (32,16)

                    data_uint8 = block.cpu().contiguous()

                    f.write(data_uint8.numpy().tobytes())

    print(
        f"Saved blocked scales to '{scales_bin_path}', "
        f"padded shape=({H},{W}), blocks=({n_block_h},{n_block_w}), "
        f"total size: {os.path.getsize(scales_bin_path)} bytes"
    )


def save_bf16_scales_blocked(scales_bf16, n_row, n_col, filename, op_name):
    """
    将 scales_bf16 (2048, 64) 转置为 (64, 2048)，然后按 16x16 元素（每块 512 字节）切分，
    按行优先顺序保存每个 block 到 bin 文件中。
    """
    n_block_h = n_row // 16
    n_block_w = n_col // 16

    block = 16

    # -----------------------------
    # 1. 计算 padding 后的尺寸
    # -----------------------------
    pad_row = math.ceil(n_row / block) * block
    pad_col = math.ceil(n_col / block) * block

    # -----------------------------
    # 2. 创建 padding tensor
    # -----------------------------
    padded = torch.zeros(
        pad_row,
        pad_col,
        dtype=scales_bf16.dtype,
        device=scales_bf16.device
    )

    # -----------------------------
    # 3. 拷贝原始数据
    # -----------------------------
    padded[:n_row, :n_col] = scales_bf16

    # -----------------------------
    # 4. 计算 block 数量
    # -----------------------------
    n_block_h = pad_row // block
    n_block_w = pad_col // block


    # 拆分为 16x16 blocks: (4, 128, 16, 16)
    scales_block_matrix = padded.view(n_block_h, 16, n_block_w, 16).permute(0, 2, 1, 3).contiguous()

    SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)
    scales_bin_path = os.path.join(save_dir, filename)

    with open(scales_bin_path, 'wb') as f:
        for i in range(n_block_h):      # 4
            for j in range(n_block_w):  # 128
                block = scales_block_matrix[i, j]  # (16, 16)
                data_uint16 = block.cpu().contiguous().view(torch.uint16)
                f.write(data_uint16.numpy().tobytes())

def save_fp8_act_scales_blocked(scales_fp8, filename, op_name):
    """
    将 scales_fp8 (128, 64) 按 16x16 分块，得到 (8, 4) 个 block。
    存储顺序：
        - 外循环：按 block 矩阵的列优先顺序（即 j 在外，i 在内）
        - 内部数据：每个 block 按列优先顺序存储（即先存第0列16个元素）
    每个 block 512 字节（16*16*2）
    """
    block_h = 16
    block_w = 32

    n_row = scales_fp8.shape[-2]
    n_col = scales_fp8.shape[-1]

     # -----------------------------------
    # 1. 计算 padding 后尺寸
    # -----------------------------------
    pad_row = math.ceil(n_row / block_h) * block_h
    pad_col = math.ceil(n_col / block_w) * block_w

    pad_h = pad_row - n_row
    pad_w = pad_col - n_col

    # -----------------------------------
    # 2. padding
    # -----------------------------------
    if scales_fp8.dim() == 2:

        padded = F.pad(scales_fp8, (0, pad_w, 0, pad_h))
        padded = padded.unsqueeze(0)  # 统一成 3D

    elif scales_fp8.dim() == 3:

        padded = F.pad(scales_fp8, (0, pad_w, 0, pad_h))

    else:
        raise ValueError("scales_fp8 must be 2D or 3D tensor")

    B, H, W = padded.shape


    n_block_h = padded.shape[0] // block_h  # 8
    n_block_w = padded.shape[1] // block_w   # 4

    # assert scales_fp8.shape == (128, 64), f"Expected (128, 64), got {scales_fp8.shape}"
    assert padded.dtype == torch.float8_e4m3fn, f"Expected fp8, got {padded.dtype}"

    # 重新 reshape 成 block 结构: (8, 16, 4, 16) -> (n_block_h, block_size, n_block_w, block_size)
    # 然后调整为 (n_block_h, n_block_w, 16, 16) 便于索引
    scales_reshaped = padded.view(n_block_h, block_h, n_block_w, block_w)
    
    # 转置为 (n_block_h, n_block_w, 16, 16)
    blocks = scales_reshaped.permute(0, 2, 1, 3).contiguous()

    SAVE_ROOT = "/root/autodl-tmp/project/Qwen3/golden_data"
    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)
    scales_bin_path = os.path.join(save_dir, filename)

    with open(scales_bin_path, 'wb') as f:
        # 外循环：按 block 矩阵的列优先顺序 (j, i)
        for b in range(B):
            for j in range(n_block_w):
                for i in range(n_block_h):
                    block = blocks[b, i, j]    # shape (16, 16), 行优先存储在内存中
                    # 要按列优先写入：转置 block → (16, 16) 列优先等价于转置后行优先
                    block_col_major = block.t().contiguous()
                    data_uint16 = block_col_major.view(torch.uint16)
                    f.write(data_uint16.cpu().numpy().tobytes())

    print(f"Saved blocked scales to '{scales_bin_path}', total size: {os.path.getsize(scales_bin_path)} bytes")

def save_fp8_weight_scales_blocked(scales_fp8, filename, op_name):
    """
    将 scales_fp8 (128, 64) 按 32x16 分块，得到 (4, 4) 个 block。
    存储顺序：
        - 外循环：按 block 矩阵的列优先顺序（即 i 在外，j 在内）
        - 内部数据：每个 block 按列优先顺序存储（即先存第0列16个元素）
    每个 block 512 字节（32*16）
    """
    block_size_h = 32
    block_size_w = 16
    n_block_h = scales_fp8.shape[0] // block_size_h  # 4
    n_block_w = scales_fp8.shape[1] // block_size_w   # 4

    # assert scales_fp8.shape == (128, 64), f"Expected (128, 64), got {scales_fp8.shape}"
    assert scales_fp8.dtype == torch.float8_e4m3fn, f"Expected fp8, got {scales_fp8.dtype}"

    # 重新 reshape 成 block 结构: (4, 32, 4, 16) -> (n_block_h, block_size, n_block_w, block_size)
    # 然后调整为 (n_block_h, n_block_w, 16, 16) 便于索引
    scales_reshaped = scales_fp8.view(n_block_h, block_size_h, n_block_w, block_size_w)
    
    # 转置为 (n_block_h, n_block_w, 32, 16)
    blocks = scales_reshaped.permute(0, 2, 1, 3).contiguous()

    SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)
    scales_bin_path = os.path.join(save_dir, filename)

    with open(scales_bin_path, 'wb') as f:
        # 外循环：按 block 矩阵的行优先顺序 (i, j)        
        for i in range(n_block_h):
            for j in range(n_block_w):
                block = blocks[i, j]    # shape (32, 16), 行优先存储在内存中
                block_col_major = block.contiguous()
                data_uint16 = block_col_major.view(torch.uint16)
                f.write(data_uint16.cpu().numpy().tobytes())

    print(f"Saved blocked scales to '{scales_bin_path}', total size: {os.path.getsize(scales_bin_path)} bytes")



# ══════════════════════════════════════════════════════════════════
# attn mask 
# ══════════════════════════════════════════════════════════════════
def save_mask_to_bin(mask_bool, filename, op_name, hw32=False):
    """
    专用函数：将 M x N 的 bool mask 矩阵保存为 .bin 文件。
    按行处理，每16个bool元素打包成一个uint16。
    假设 N % 16 == 0。
    """
    SAVE_ROOT = "/root/autodl-tmp/project/golden_data"
    save_dir = os.path.join(SAVE_ROOT, op_name)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    mask_uint8 = mask_bool.cpu().contiguous().to(torch.uint8) # Shape: [M, N]

    M, N = mask_uint8.shape
    num_uint16_per_row = N // 16
    total_uint16 = M * num_uint16_per_row

    # 创建用于存储打包后数据的 numpy 数组
    import numpy as np
    packed_mask = np.zeros(total_uint16, dtype=np.uint16)

    # 按行、按每16个元素进行打包
    for i in range(M):
        row = mask_uint8[i] # 取第 i 行, shape: [N]
        for j in range(num_uint16_per_row):
            # 取当前块的16个元素 (从高位到低位打包)
            block = row[j*16 : (j+1)*16] # shape: [16]
            # 将16个bit打包成一个uint16 (索引0是最高位MSB, 索引15是最低位LSB)
            # 方法: 累加 block[k] * (2^(15-k))
            uint16_val = 0
            for k in range(16):
                if block[k]:
                    uint16_val |= (1 << k)

            packed_mask[i * num_uint16_per_row + j] = uint16_val

    # 保存为二进制文件
    if hw32:
        packed = torch.Tensor(packed_mask.reshape(M, -1)).to(torch.uint16).reshape(int(M/128), 128, -1)
        act_slicehw_layout(packed, filename, op_name, block2=8)
    else:
        packed_mask.tofile(filepath)
    print(f"Saved mask {mask_bool.shape} to '{filepath}' as {total_uint16} uint16 values (packed)")