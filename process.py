import torch
import torch.nn.functional as F

def split_tensor_n_dim(tensor, num_split=4, cut_dim = 1, intervals = None):
    """
    按 N 维 (dim=1) 切分 tensor
    input:  [M, N]
    output: list of [M, N/num_split]
    
    """
    if intervals is None and tensor.shape[cut_dim] % num_split == 0:
        return torch.chunk(tensor, num_split, dim=cut_dim)
    elif intervals is not None:
        slices = []
        for start, end in intervals:
            # 构建切片对象：对于 dim 维度取 [start:end]，其他维度取全部
            idx = [slice(None)] * tensor.ndim
            idx[cut_dim] = slice(start, end)
            slices.append(tensor[tuple(idx)])
        return slices
    else :
        raise ValueError("tensor.shape[cut_dim] % num_split != 0 and (group_size > 0 and chunk_size > 0) is not supported")
    

def padding_token(tensor, token_idx, max_tokens=256, pad_right=32):
    """
    处理 张量：创建零张量、索引复制、填充，并保存为二进制文件。
    Args:
        tensor: 输入张量。
        token_idx: 索引复制时使用的索引张量或列表。
        expert_idx: 用于文件名的专家索引。
        layer_index: 用于文件名的层索引。
        op_name: 传递给 save_tensor_to_bin 的操作名称。
        max_tokens: 零张量的行数（默认 256）。
        pad_right: 最后一维右侧填充的列数（默认 32）。
    """
    # 创建形状为 (max_tokens, 列数) 的零张量
    zeros = torch.zeros((max_tokens, tensor.shape[-1]),
                           dtype=tensor.dtype,
                           device=tensor.device)
    # 根据 token_idx 将 up_tensor 的值复制到 up_zeros 的对应行
    zeros.index_copy_(0, token_idx, tensor)
    # 在最后一维右侧填充 pad_right 个零
    pad = F.pad(zeros, (0, pad_right, 0, 0))
    # 保存张量到二进制文件
    return pad

def reorder_for_rope(tensor):
    """
    对张量的最后一维进行 RoPE 重排序（交错前后半段）。
    """
    device = tensor.device
    d = tensor.shape[-1]
    half = d // 2
    indices = torch.arange(d, device=device).reshape(2, half).t().reshape(-1)
    return tensor.index_select(-1, indices)


def reshape_reorder_weight(weight, head_dim):
    """
    对多头注意力权重进行 reshape → 重排序 → reshape 恢复。

    假设输入 weight 形状为 (num_heads * head_dim, d_model)，
    内部将其重塑为 (num_heads, d_model, head_dim)，重排序最后一维，再恢复原始形状。

    Args:
        weight (torch.Tensor): 原始权重张量，形状 (num_heads * head_dim, d_model)
        head_dim (int): 每个头的维度

    Returns:
        torch.Tensor: 重排序后的权重，形状与输入相同
    """
    P = weight.shape[0]                     # num_heads * head_dim
    # 重塑为 (P, -1, head_dim)，-1 自动推断为 d_model
    weight_3d = weight.reshape(P, -1, head_dim)
    weight_reordered = reorder_for_rope(weight_3d)
    # 恢复原始形状
    return weight_reordered.reshape(weight.shape)


def reshape_reorder_bias(bias, head_dim):
    """
    对多头注意力偏置进行 reshape → 重排序 → reshape 恢复。

    假设输入 bias 形状为 (num_heads * head_dim,)，
    内部将其重塑为 (num_heads, head_dim)，重排序最后一维，再恢复一维形状。

    Args:
        bias (torch.Tensor): 原始偏置张量，形状 (num_heads * head_dim,)
        head_dim (int): 每个头的维度

    Returns:
        torch.Tensor: 重排序后的偏置，形状与输入相同
    """
    # 重塑为 (num_heads, head_dim)
    bias_2d = bias.reshape(-1, head_dim)
    bias_reordered = reorder_for_rope(bias_2d)
    # 恢复一维形状
    return bias_reordered.reshape(-1)