
import torch
from torch import nn
import numpy as np
import auto_round

def unpack_tensor_with_torch(MulLinear, packed_tensor):
    target_dtype = torch.int8 if not hasattr(MulLinear, "qzeros") or "int" not in MulLinear.dtype else torch.uint8
    target_len = packed_tensor.shape[1] * MulLinear.n_pack
    unpacked_tensor = torch.zeros(packed_tensor.shape[0], target_len, dtype=target_dtype).to(packed_tensor.device)
    mask = torch.tensor(2**MulLinear.bits - 1, dtype=MulLinear.compression_dtype).to(packed_tensor.device)
    for j in range(packed_tensor.shape[1]):
        for e in range(MulLinear.n_pack):
            index = j * MulLinear.n_pack + e
            tmp = packed_tensor[:, j]
            tmp = tmp << (MulLinear.compress_bits - MulLinear.bits * (e + 1))
            tmp = tmp >> MulLinear.compress_bits - MulLinear.bits
            if target_dtype == torch.uint8:
                tmp &= mask  # remove sign bit
            unpacked_tensor[:, index].copy_(tmp.type(target_dtype))
    # logger.info(f"*****{unpacked_tensor}")
    return unpacked_tensor

def unpack_tensor_with_numpy(MulLinear, packed_tensor):
    packed_array = packed_tensor.cpu().numpy()
    target_dtype = np.int8 if not hasattr(MulLinear, "qzeros") or "int" not in MulLinear.dtype else np.uint8
    target_len = packed_array.shape[1] * MulLinear.n_pack
    unpacked_array = np.zeros((packed_array.shape[0], target_len), dtype=target_dtype)
    mask = np.uint8(2**MulLinear.bits - 1)
    for j in range(packed_array.shape[1]):
        for e in range(MulLinear.n_pack):
            index = j * MulLinear.n_pack + e
            tmp = packed_array[:, j]
            tmp = np.left_shift(tmp, MulLinear.compress_bits - MulLinear.bits * (e + 1))
            tmp = np.right_shift(tmp, MulLinear.compress_bits - MulLinear.bits)
            if target_dtype == np.uint8:
                tmp &= mask
            unpacked_array[:, index] = tmp.astype(target_dtype)
    unpacked_tensor = torch.from_numpy(unpacked_array).to(device=packed_tensor.device)
    return unpacked_tensor

def unpack_tensor(MulLinear, packed_tensor):
    if packed_tensor.device.type == "cuda":
        return unpack_tensor_with_torch(MulLinear, packed_tensor)
    else:
        return unpack_tensor_with_numpy(MulLinear, packed_tensor)


def recover(self, MulLinear, sym_quant_ind):
    if self.dtype == "nvfp4" or self.dtype == "nv_fp":
            from auto_round.data_type.nvfp import get_reciprocal
            from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
            qweight, orig_shape, pad_len = reshape_pad_tensor_by_group_size(self.qweight, self.groupsize)
            assert qweight.ndim == 2
            m, n = qweight.shape
            fp8_weight =  revert_tensor_by_pad(qweight.reshape(m, n), orig_shape=orig_shape, pad_len=pad_len)
            return fp8_weight


    # logger.debug(f"Recovering {self} weight")
    scales = MulLinear.scales.T.contiguous() if MulLinear.use_optimum_format else MulLinear.scales
    qweight = MulLinear.qweight.T.contiguous() if MulLinear.use_optimum_format else MulLinear.qweight

    device = scales.device
    # MulLinear = MulLinear.to(device)
    # fp32_weight = torch.zeros(self.out_features, self.in_features, dtype=self.float_type).to(device)
    fp32_weight = torch.zeros(MulLinear.out_features, MulLinear.in_features, dtype=torch.float32).to(device)
    if MulLinear.g_idx is None:
        # used for recovering fp32_weight
        MulLinear.g_idx = torch.tensor([i // MulLinear.groupsize for i in range(MulLinear.in_features)], dtype=torch.int32)
    # unpack weight
    if not MulLinear.use_optimum_format and MulLinear.compression_dim == 0:
        qweight = qweight.T.contiguous()
    weight = unpack_tensor(MulLinear, qweight)
    weight_int4 = torch.zeros(MulLinear.out_features, MulLinear.in_features, dtype=torch.int8).to(device)
    scales_bf16 = torch.zeros(MulLinear.out_features, MulLinear.in_features, dtype=torch.bfloat16).to(device)
    if not MulLinear.use_optimum_format and MulLinear.compression_dim == 0:
        weight = weight.T.contiguous()
    weight = weight[: MulLinear.out_features, : MulLinear.in_features]  # avoid oversize
    if "int" not in MulLinear.dtype:
        new_weight = torch.zeros(MulLinear.out_features, MulLinear.in_features).to(device)
        for k, v in self.int2float_mapping.items():
            new_weight += torch.where(weight == k, v, 0)
        weight = new_weight
    # unpack zero_point
    if hasattr(MulLinear, "qzeros"):
        qzeros = MulLinear.qzeros.T.contiguous() if MulLinear.use_optimum_format else MulLinear.qzeros
        if MulLinear.use_optimum_format or MulLinear.compression_dim == 0:
            qzeros = qzeros.T.contiguous()
        zp = unpack_tensor(MulLinear, qzeros)
        if MulLinear.use_optimum_format or MulLinear.compression_dim == 0:
            zp = zp.T.contiguous()
        zp = zp[: scales.shape[0], : scales.shape[1]]  # avoid oversize
        if MulLinear.use_optimum_format:
            # zp -= 1 may cause zp == -1, after recover it becomes 2**self.bits - 1
            zp += 1
            zp = torch.where(zp > (2**MulLinear.bits - 1), 0, zp)
        # recover fp32 weight with int_weight, scale, and zero_point
        for idx in range(MulLinear.in_features):
            if sym_quant_ind == True:
                weight_int4[:, idx] = torch.subtract(weight[:, idx], zp[:, MulLinear.g_idx[idx]]).to(torch.int8)
                bias = None
            else:
                weight_int4[:, idx] = (weight[:, idx] - 8).to(torch.int8)
                bias = ((zp - 8).to(torch.int8)) * scales

            # scales_bf16[:, idx] = scales[:, MulLinear.g_idx[idx]]
            fp32_weight[:, idx] = (torch.subtract(weight[:, idx], zp[:, MulLinear.g_idx[idx]]).to(torch.int8)) * scales[
                :, MulLinear.g_idx[idx]
            ]
    else:
        # recover fp32 weight with int_weight, scale
        for idx in range(MulLinear.in_features):
            fp32_weight[:, idx] = weight[:, idx] * scales[:, MulLinear.g_idx[idx]]
    
    weight_debug = weight_int4 * scales_bf16

    return weight_int4, scales, bias, fp32_weight