import torch
import os
import numpy as np
from torch.nn import functional as F
import math


def get_dtype(dtype):
    data_size = 0
    if dtype == torch.float8_e4m3fn:
        np_dtype = np.uint8
        data_size = 1
    elif dtype == torch.bfloat16:
        np_dtype = np.uint16
        data_size = 2
    elif dtype == torch.int8:
        np_dtype = np.int8
        data_size = 1
    elif dtype == torch.uint8:
        np_dtype = np.uint8
        data_size = 1
    elif dtype == torch.float32:
        np_dtype = np.float32
        data_size = 4
    elif dtype == torch.int32:
        np_dtype = np.int32
        data_size = 4
    else:
        raise(ValueError(f"Unsupported dtype: {dtype}"))
    return np_dtype, data_size

def HW_TO_ddr(tensor, dim, block_size, block_W, block_H):
    # 1. 计算块的数量
    n_block_m = dim[0] // block_H
    n_block_k = dim[1] // block_W
    res = torch.zeros(dim, dtype=torch.bfloat16)
    # 2. 遍历进行还原
    for j in range(n_block_k):
        for i in range(n_block_m):
            res[i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W] = tensor[(j*n_block_m+i)*block_size:(j*n_block_m+i+1)*block_size].view(block_H, block_W)
    
    return res

def WH_TO_ddr(tensor, dim, block_size, block_W, block_H):
    # 1. 计算块的数量
    n_block_m = dim[0] // block_H
    n_block_k = dim[1] // block_W
    res = torch.zeros(dim, dtype=torch.bfloat16)
    # 2. 遍历进行还原
    for i in range(n_block_m):     
        for j in range(n_block_k):  
            block = tensor[(i*n_block_k+j)*block_size:(i*n_block_k+j+1)*block_size]
            block = block.view(block_H, block_W)
            block = block.transpose(0, 1)
            res[i*block_H:(i+1)*block_H, j*block_W:(j+1)*block_W] = block
    return res


def slice_HW_compare(origin_tensor, compare_tensor, block_size, block_W, block_H):
    """
    将原始的tensor切换为块，用于比较。
    block_W: 块宽度
    block_H: 块高度
    """
    n_block_m = origin_tensor.shape[0] // block_H
    n_block_k = origin_tensor.shape[1] // block_W

    # 拆分为 block_H x block_W的blocks: (4, 128, 16, 16)
    scales_block_matrix = origin_tensor.view(n_block_m, block_H, n_block_k, block_W).permute(0, 2, 1, 3).contiguous()
    # 进行比较
    for j in range(n_block_k):      # 4
        for i in range(n_block_m):  # 128
            block = scales_block_matrix[i, j]  # (16, 16)
            block = block.reshape(-1)
            # 从compare_tensor中取出对应的块
            compare_block = compare_tensor[(j*n_block_m+i)*block_size:(j*n_block_m+i+1)*block_size] 
            # 或用torch.all(block==compare_block),  torch.allclose(block, compare_block, rtol=1e-3, atol=1e-3)
            if not torch.all(block==compare_block):
                print(f"第{j*n_block_m+i}个块不一致")
                return False
    return True

def slice_WH_compare(origin_tensor, compare_tensor, block_size, block_W, block_H, is_transpose=True):
    
    n_block_m = origin_tensor.shape[0] // block_H
    n_block_k = origin_tensor.shape[1] // block_W

    # 拆分为 block_H x block_W的blocks: 
    scales_block_matrix = origin_tensor.view(n_block_m, block_H, n_block_k, block_W).permute(0, 2, 1, 3).contiguous()
    # 进行比较
    for i in range(n_block_m):      # 4
        for j in range(n_block_k):  # 128
            block = scales_block_matrix[i, j]  # (16, 16)
            if is_transpose:
                block = block.transpose(0, 1)   
            block = block.reshape(-1)
            # 从compare_tensor中取出对应的块
            compare_block = compare_tensor[(i*n_block_k+j)*block_size:(i*n_block_k+j+1)*block_size] 
            if not torch.all(block == compare_block):
                print(f"第{i*n_block_k+j}个块不一致")
                return False
    return True


def act_matrix_layout(act_tensor, SAVE_ROOT, filename, op_name):
    """
    将 scales_bf16 (256, 1024) ，然后按 16x16 元素（每块 512 字节）切分，
    按行优先顺序保存每个 block 到 bin 文件中。
    """
    if len(act_tensor.shape) == 3:
        B, M, K = act_tensor.shape
        act_tensor  = act_tensor.reshape(B*M, K)
        M = B*M
    elif len(act_tensor.shape) == 2:
        M, K = act_tensor.shape
    else:
        raise ValueError("act_tensor must be 2D or 3D tensor")

    block = 16

    # -----------------------------
    # 1. 计算 padding 后的尺寸
    # -----------------------------
    pad_row = math.ceil(M / block) * block
    pad_col = math.ceil(K / block) * block

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
    n_block_m = pad_row // block
    n_block_k = pad_col // block


    # 拆分为 16x16 blocks: (4, 128, 16, 16)
    scales_block_matrix = padded.view(n_block_m, 16, n_block_k, 16).permute(0, 2, 1, 3).contiguous()

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

def read_act_matrix_layout(filename, original_shape, block_size=16):
    """
    从分块存储的文件中恢复原始张量
    original_shape: 原始张量形状 (M, K) 或 (B, M, K)
    """
    M, K = original_shape[-2], original_shape[-1]

    # 计算填充后尺寸
    pad_row = math.ceil(M / block_size) * block_size
    pad_col = math.ceil(K / block_size) * block_size
    n_block_m = pad_row // block_size
    n_block_k = pad_col // block_size

    # 创建空张量
    padded = torch.zeros(pad_row, pad_col, dtype=torch.bfloat16)

    with open(filename, 'rb') as f:
        for j in range(n_block_k):      # 列块
            for i in range(n_block_m):  # 行块
                # 读取一个块的数据
                data_bytes = f.read(block_size * block_size * 2)  # 512字节
                if not data_bytes:
                    break

                # 转换为tensor
                block_np = np.frombuffer(data_bytes, dtype=np.uint16)
                block = torch.tensor(block_np, dtype=torch.uint16)
                block = block.view(torch.bfloat16).reshape(block_size, block_size)

                # 放回对应位置
                row_start = i * block_size
                col_start = j * block_size
                padded[row_start:row_start+block_size, 
                       col_start:col_start+block_size] = block

    # 去除填充部分
    restored = padded[:M, :K]

    # 恢复原始形状
    if len(original_shape) == 3:
        B, M, K = original_shape
        restored = restored.reshape(B, M, K)

    return restored


def save_tensor_to_bin(tensor, SAVE_ROOT, filename, op_name):
    """
    保存 tensor 为 .bin 文件（原始字节）：
        - bf16 → uint16（按位 reinterpret）
        - fp8  → uint8（按位 reinterpret）
        - int8 → int8（直接保存）
    """

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


class golden_data_generator:
    def __init__(self, SAVE_ROOT="golden_data"):
        self.SAVE_ROOT = SAVE_ROOT

    
    def element_add(self, input_size, output_size, dtype:torch.dtype, way="l1buf"):
        """ Args:
            input_size: 输入维度的大小
            output_size: 输出维度的大小
            dtyde：数据类型
            way：保存方式：l1buf or ddr

        
        """
        input1 = torch.randn((input_size, output_size), dtype=dtype)  
        input2 = torch.randn((input_size, output_size), dtype=dtype)
        output = input1 + input2
        if dtype == torch.bfloat16:
            dtype_name = "bf16"
        elif dtype == torch.float32:
            dtype_name = "f32"
        else:
            dtype_name = str(dtype).split(".")[-1]
        if way == "l1buf":
            act_matrix_layout(input1, self.SAVE_ROOT, f"element_add_input_1_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(input2, self.SAVE_ROOT, f"element_add_input_2_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output, self.SAVE_ROOT, f"element_add_output_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
        elif way == "ddr":
            # ddr方式保存，没对齐的话，需要手动进行padding
            save_tensor_to_bin(input1, self.SAVE_ROOT, f"element_add_input_1_{input_size}_{output_size}_{dtype_name}.bin", "identity")
            save_tensor_to_bin(input2, self.SAVE_ROOT, f"element_add_input_2_{input_size}_{output_size}_{dtype_name}.bin", "identity")
            save_tensor_to_bin(output, self.SAVE_ROOT, f"element_add_output_{input_size}_{output_size}_{dtype_name}.bin", "identity")
        else:
            raise(ValueError(f"Unsupported way: {way}"))

    def element_add_4in_5out(self, input_size, output_size, dtype:torch.dtype, way="l1buf"):
        """
        需求：在l1buf中是sliceHW32 排布，bf16格式，
        输入：四个输入box，分别是input0、input1、input2、input3、box size都是[256,2880], 
        输出：输入box对应的element相加，得到新的box[256,2880]
        输出共5个：output02 = input0 + input2、output13 = input1+input3 、output21 = input2 + input1、output30 = input3 + input30、
        output0123 = input0 + input1 + input2 + input3
        """
        
        input0 = torch.randn((input_size, output_size), dtype=dtype)  
        input1 = torch.randn((input_size, output_size), dtype=dtype)
        input2 = torch.randn((input_size, output_size), dtype=dtype)
        input3 = torch.randn((input_size, output_size), dtype=dtype)
        output02 = input0 + input2
        output13 = input1 + input3
        output21 = input2 + input1
        output30 = input3 + input0
        output0123 = input0 + input1 + input2 + input3
        if dtype == torch.bfloat16:
            dtype_name = "bf16"
        elif dtype == torch.float32:
            dtype_name = "f32"
        else:
            dtype_name = str(dtype).split(".")[-1]
        if way == "l1buf":
            act_matrix_layout(input0, self.SAVE_ROOT, f"element_input0_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(input1, self.SAVE_ROOT, f"element_input1_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(input2, self.SAVE_ROOT, f"element_input2_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(input3, self.SAVE_ROOT, f"element_input3_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output02, self.SAVE_ROOT, f"element_output02_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output13, self.SAVE_ROOT, f"element_output13_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output21, self.SAVE_ROOT, f"element_output21_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output30, self.SAVE_ROOT, f"element_output30_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
            act_matrix_layout(output0123, self.SAVE_ROOT, f"element_output0123_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")

        elif way == "ddr":
            # ddr方式保存，没对齐的话，需要手动进行padding
            save_tensor_to_bin(input0, self.SAVE_ROOT, f"element_input0_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(input1, self.SAVE_ROOT, f"element_input1_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(input2, self.SAVE_ROOT, f"element_input2_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(input3, self.SAVE_ROOT, f"element_input3_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(output02, self.SAVE_ROOT, f"element_output02_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(output13, self.SAVE_ROOT, f"element_output13_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(output21, self.SAVE_ROOT, f"element_output21_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(output30, self.SAVE_ROOT, f"element_output30_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            save_tensor_to_bin(output0123, self.SAVE_ROOT, f"element_output0123_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
        else:
            raise(ValueError(f"Unsupported way: {way}"))

    def four_element_4in_9out(self, input_size, output_size, dtype: torch.dtype, way="l1buf"):
            """
            需求：在l1buf中是sliceHW32 排布，bf16格式，
            输入：四个输入box，分别是input0、input1、input2、input3、box size都是[256,2880], 
            输出：输入box对应的element相加，得到新的box[256,2880]
            输出共9个：output01 = input0 + input1、output10 = input1+input0 、output23 = input2 + input3、output32 = input3 + input2、
            output_die0=output01 + output32 、output_die1 = output10 + output23 、output_die2 = output23 + output10 、output_die3 =  output32 + output01 、
            output0123 = input0 + input1 + input2 + input3
            """
            
            input0 = torch.randn((input_size, output_size), dtype=dtype)  
            input1 = torch.randn((input_size, output_size), dtype=dtype)
            input2 = torch.randn((input_size, output_size), dtype=dtype)
            input3 = torch.randn((input_size, output_size), dtype=dtype)
            
            # 计算需求中明确要求的9个输出
            output01 = input0 + input1
            output10 = input1 + input0
            output23 = input2 + input3
            output32 = input3 + input2
            output_die0 = output01 + output32
            output_die1 = output10 + output23
            output_die2 = output23 + output10
            output_die3 = output32 + output01
            output0123 = input0 + input1 + input2 + input3
            
            if dtype == torch.bfloat16:
                dtype_name = "bf16"
            elif dtype == torch.float32:
                dtype_name = "f32"
            else:
                dtype_name = str(dtype).split(".")[-1]
            if way == "l1buf":
                act_matrix_layout(input0, self.SAVE_ROOT, f"element_input0_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(input1, self.SAVE_ROOT, f"element_input1_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(input2, self.SAVE_ROOT, f"element_input2_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(input3, self.SAVE_ROOT, f"element_input3_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output01, self.SAVE_ROOT, f"element_output01_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output10, self.SAVE_ROOT, f"element_output10_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output23, self.SAVE_ROOT, f"element_output23_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output32, self.SAVE_ROOT, f"element_output32_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output_die0, self.SAVE_ROOT, f"element_output_die0_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output_die1, self.SAVE_ROOT, f"element_output_die1_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output_die2, self.SAVE_ROOT, f"element_output_die2_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output_die3, self.SAVE_ROOT, f"element_output_die3_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")
                act_matrix_layout(output0123, self.SAVE_ROOT, f"element_output0123_{input_size}_{output_size}_{dtype_name}_l1buf.bin", "L1buf_layout")

            elif way == "ddr":
                # ddr方式保存，没对齐的话，需要手动进行padding
                save_tensor_to_bin(input0, self.SAVE_ROOT, f"element_input0_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(input1, self.SAVE_ROOT, f"element_input1_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(input2, self.SAVE_ROOT, f"element_input2_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(input3, self.SAVE_ROOT, f"element_input3_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output01, self.SAVE_ROOT, f"element_output01_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output10, self.SAVE_ROOT, f"element_output10_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output23, self.SAVE_ROOT, f"element_output23_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output32, self.SAVE_ROOT, f"element_output32_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output_die0, self.SAVE_ROOT, f"element_output_die0_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output_die1, self.SAVE_ROOT, f"element_output_die1_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output_die2, self.SAVE_ROOT, f"element_output_die2_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output_die3, self.SAVE_ROOT, f"element_output_die3_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
                save_tensor_to_bin(output0123, self.SAVE_ROOT, f"element_output0123_{input_size}_{output_size}_{dtype_name}_ddr.bin", "identity")
            else:
                raise(ValueError(f"Unsupported way: {way}"))

    def element_add_test(self, file_input1, file_input2, file_output, dtype:torch.dtype):
        np_dtype, _ = get_dtype(dtype)

        input1 = torch.tensor(np.fromfile(file_input1, dtype=np_dtype)).view(dtype=dtype)
        input2 = torch.tensor(np.fromfile(file_input2, dtype=np_dtype)).view(dtype=dtype)
        output = torch.tensor(np.fromfile(file_output, dtype=np_dtype)).view(dtype=dtype)

        if torch.all(input1+input2 == output):
            print("input1+input2 == output")
        else:
            print("input1+input2 != output")

    def four_element_add_test(self, file_input1, file_input2, file_input3, file_input4, file_output, dtype:torch.dtype):
        np_dtype, _ = get_dtype(dtype)

        input1 = torch.tensor(np.fromfile(file_input1, dtype=np_dtype)).view(dtype=dtype)
        input2 = torch.tensor(np.fromfile(file_input2, dtype=np_dtype)).view(dtype=dtype)
        input3 = torch.tensor(np.fromfile(file_input3, dtype=np_dtype)).view(dtype=dtype)
        input4 = torch.tensor(np.fromfile(file_input4, dtype=np_dtype)).view(dtype=dtype)
        output = torch.tensor(np.fromfile(file_output, dtype=np_dtype)).view(dtype=dtype)

        if not torch.all(input1+input2+input3+input4 == output):
            print("input1+input2+input3+input4 != output")
        else:
            print("input1+input2+input3+input4 == output")

    def rope_test(self, file_data, file_weight, file_output, dtype:torch.dtype, compare_size):
        np_dtype, _ = get_dtype(dtype)
        
        data = torch.tensor(np.fromfile(file_data, dtype=np_dtype)).view(dtype=dtype)
        weight = torch.tensor(np.fromfile(file_weight, dtype=np_dtype)).view(dtype=dtype)
        output = torch.tensor(np.fromfile(file_output, dtype=np_dtype)).view(dtype=dtype)
        
        flag = True
        for i in range(0, compare_size, 2):
            if((data[i] * weight[i+1] - data[i+1] * weight[i]) != output[i]):
                print(f"output的第{i}个数据的第1个计算，输出结果不一致")
                flag = False
                break
            if((data[i+1] * weight[i+1] + data[i] * weight[i]) != output[i+1]):
                print(f"output的第{i}个数据的第2个计算，结果不一致")
                flag = False
                break
        if flag:
            print("rope计算结果一致")

    def slice_sliceWH32_generator(self, tensor, dtype:torch.dtype, name):
        _, data_size = get_dtype(dtype)
        block_size = 512//data_size
        block_H = 32//data_size
        block_W = block_size//block_H
        n_block_m = tensor.shape[0] // block_H
        n_block_k = tensor.shape[1] // block_W
        #方法一：
        # 先保存ddr
        save_tensor_to_bin(tensor, self.SAVE_ROOT, f"{name}_{tensor.shape[0]}_{tensor.shape[1]}_bf16_ddr.bin", "identity")
        
        block_matrix = tensor.view(n_block_m, block_H, n_block_k, block_W).permute(0, 2, 1, 3).contiguous()

        save_dir = os.path.join(self.SAVE_ROOT, "L1buf_layout")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{name}_{tensor.shape[0]}_{tensor.shape[1]}_bf16_l1buf.bin"
        bin_path = os.path.join(save_dir, file_name)

        with open(bin_path, 'wb') as f:
            for i in range(n_block_m):      
                for j in range(n_block_k):  
                    block = block_matrix[i, j]
                    # 转置
                    block = block.transpose(0, 1)
                    data_uint16 = block.cpu().contiguous().view(torch.uint16)
                    f.write(data_uint16.numpy().tobytes())

        print(f"Saved blocked scales to '{bin_path}', total size: {os.path.getsize(bin_path)} bytes")

    def slice_sliceHW32_generator(self, input_size, output_size, dtype:torch.dtype, name, tensor=None):
        if tensor == None:
            tensor = torch.randn((input_size, output_size), dtype=dtype)
        save_tensor_to_bin(tensor, self.SAVE_ROOT, f"{name}_{input_size}_{output_size}_{dtype}_ddr.bin", "identity")
        act_matrix_layout(tensor, self.SAVE_ROOT, f"{name}_{input_size}_{output_size}_{dtype}_l1buf.bin", "L1buf_layout")

    def detect_dim_align(self, origin_file, data_file, dtype:torch.dtype, dim, align_way, align_size):
        np_dtype, data_size = get_dtype(dtype)
        origin = torch.tensor(np.fromfile(origin_file, dtype=np_dtype)).view(dtype=dtype)
        origin = origin.reshape(dim)
        data = torch.tensor(np.fromfile(data_file, dtype=np_dtype)).view(dtype=dtype)
        # 1. 首先判断数据维度是否相同
        data_dim = origin.numel()
        if data.numel() == data_dim:
            print(f"两个文件的shape一致，数据大小为{data_dim}")    
        else:
            print(f"两个文件的shape不一致，ddr文件大小为{data.numel()}，l1buf文件大小为{data_dim}")
            return False
        # 2. 检查对齐方式
        if align_way == "sliceHW":
            # 3. 检测数据维度是否为32B, 128B, 256B，512B对齐
            # 检测数据维度
            if align_size == "H16W16B":
                if data_dim*data_size % 16 == 0:
                    # block_size为每个块的占用多个数据，block_W为32B维度上的元素数量，block_H为block_size/block_W
                    block_size = 16 * 16//data_size
                    block_W = 16//data_size
                    block_H = 16
                    #对比两个向量
                    if len(dim) == 2:
                        res = slice_HW_compare(origin, data, block_size, block_W, block_H)
                        if res:
                            print(f"W方向是 16B 对齐")
                        else:
                            print(f"W方向不是 16B 对齐")
                            return False
                    elif len(dim) == 3:
                        # dim[0]
                        for i in range(dim[0]):
                            print(f"{i}. 正在检测[{i},:,:]的数据")
                            origin_block = origin[i]
                            data_block = data[i*dim[1]*dim[2]:(i+1)*dim[1]*dim[2]]
                            res = slice_HW_compare(origin_block, data_block, block_size, block_W, block_H)
                            if not res:
                                print(f"[{i},:,:]在W方向16B不对齐")
                                return False
                            else:
                                print(f"[{i},:,:]在W方向16B对齐")   
                        print(f"三维矩阵在W方向16B对齐")
                    else:
                        raise ValueError(f"Unsupported dim: {dim}")
                else:
                    print(f"文件大小为{data_dim*data_size}B,不能整除16B")
                    return False

            elif align_size == "32B":
                if data_dim*data_size % 32 == 0:
                    # block_size为每个块的占用多个数据，block_W为32B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 32//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    if len(dim) == 2:
                        res = slice_HW_compare(origin, data, block_size, block_W, block_H)
                        if res:
                            print(f"W方向是 32B 对齐")
                        else:
                            print(f"W方向不是 32B 对齐")
                            return False
                    elif len(dim) == 3:
                        # dim[0]
                        for i in range(dim[0]):
                            print(f"{i}. 正在检测[{i},:,:]的数据")
                            origin_block = origin[i]
                            data_block = data[i*dim[1]*dim[2]:(i+1)*dim[1]*dim[2]]
                            res = slice_HW_compare(origin_block, data_block, block_size, block_W, block_H)
                            if not res:
                                print(f"[{i},:,:]在W方向32B不对齐")
                                return False
                            else:
                                print(f"[{i},:,:]在W方向32B对齐")   
                        print(f"三维矩阵在W方向32B对齐")
                    else:
                        raise ValueError(f"Unsupported dim: {dim}")
                else:
                    print(f"文件大小为{data_dim*data_size}B,不能整除32B")
                    return False
            elif align_size == "128B":
                if data_dim*data_size % 128 == 0:
                    # block_size为每个块的总大小，block_W为128B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 128//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_HW_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在W方向是128B对齐")
                    else:
                        print(f"数据在W方向128B不对齐")
                        return False
                else:
                    print(f"数据大小为{data_dim*data_size}B,不能整除128B")
                    return False
            elif align_size == "256B":
                if data_dim*data_size % 256 == 0:
                    # block_size为每个块的总大小，block_W为256B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 256//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_HW_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在W方向是256B对齐")
                    else:
                        print(f"数据在W方向256B不对齐")
                        return False
                else:
                    print(f"数据大小为{data_dim*data_size}B,不能整除256B")
                    return False
            elif align_size == "512B":
                if data_dim*data_size % 256 == 0:
                    # block_size为每个块的总大小，block_W为256B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 512//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_HW_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在W方向是512B对齐")
                    else:
                        print(f"数据在W方向512B不对齐")
                        return False
                else:
                    print(f"数据大小为{data_dim*data_size}B,不能整除512B")
                    return False
            else:
                print(f"Unsupported align_size: {align_size}")
                return False
        elif align_way == "sliceWH":
            # 检测数据维度是否为512B，256B，128B, 32B对齐
            if align_size == "scale_W16H32B":
                if data_dim*data_size % 32 == 0:
                    # block_size为每个块的占用多个数据，block_W为32B维度上的元素数量，block_H为block_size/block_W
                    block_size = 16 * 32//data_size
                    block_W = 16
                    block_H = 32//data_size
                    #对比两个向量
                    if len(dim) == 2:
                        res = slice_WH_compare(origin, data, block_size, block_W, block_H, is_transpose=False)
                        if res:
                            print(f"H方向是 32B 对齐")
                        else:
                            print(f"H方向不是 32B 对齐")
                            return False
                    elif len(dim) == 3:
                        # dim[0]
                        for i in range(dim[0]):
                            print(f"{i}. 正在检测[{i},:,:]的数据")
                            origin_block = origin[i]
                            data_block = data[i*dim[1]*dim[2]:(i+1)*dim[1]*dim[2]]
                            res = slice_WH_compare(origin_block, data_block, block_size, block_W, block_H)
                            if not res:
                                print(f"[{i},:,:]在H方向32B不对齐")
                                return False
                            else:
                                print(f"[{i},:,:]在H方向32B对齐")   
                        print(f"三维矩阵在H方向32B对齐")
                    else:
                        raise ValueError(f"Unsupported dim: {dim}")
                else:
                    print(f"文件大小为{data_dim*data_size}B,不能整除32B")
                    return False
            elif align_size == "32B":
                if data_dim*data_size % 32 == 0:
                    # block_size为每个块的总大小，block_W为32B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 32//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_WH_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在H方向是32B对齐")
                    else:
                        print(f"数据在H方向是32B不对齐")
                        return False
                else:
                    print(f"数据大小为{data_dim*data_size}B,不能整除32B")
                    return False
            elif align_size == "128B":
                if data_dim*data_size % 128 == 0:
                    # block_size为每个块的总大小，block_W为128B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 128//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_WH_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在H方向是128B对齐")
                    else:
                        print(f"数据在H方向是128B不对齐")
                        return False
                else:
                    print(f"数据大小为{data_dim*data_size}B,不能整除128B")
                    return False
            elif align_size == "256B":
                if data_dim*data_size % 256 == 0:
                    # block_size为每个块的总大小，block_W为256B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 256//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_WH_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在H方向是256B对齐")
                    else:
                        print(f"数据在H方向是256B不对齐")
                        return False
            elif align_size == "512B":
                if data_dim*data_size % 512 == 0:
                    # block_size为每个块的总大小，block_W为512B维度上的元素数量，block_H为block_size/block_W
                    block_size = 512//data_size
                    block_W = 512//data_size
                    block_H = block_size//block_W
                    #对比两个向量
                    res = slice_WH_compare(origin, data, block_size, block_W, block_H)
                    if res:
                        print(f"数据在H方向是512B对齐")
                    else:
                        print(f"数据在H方向是512B不对齐")
                        return False
            else:
                print(f"数据大小为{data_dim*data_size}B,不能整除512B")
                return False
        else:
            raise(ValueError(f"Unsupported align_way: {align_way}"))
        return True
    def stack_test(self, file_x1, file_x2, file_x3, dtype:torch.dtype, dim):
        np_dtype, _ = get_dtype(dtype)
        x1 = torch.tensor(np.fromfile(file_x1, dtype=np_dtype)).view(dtype=dtype)
        x2 = torch.tensor(np.fromfile(file_x2, dtype=np_dtype)).view(dtype=dtype)
        x3 = torch.tensor(np.fromfile(file_x3, dtype=np_dtype)).view(dtype=dtype)
        
        for i in range(dim):
            if x1[i] != x3[i]:
                print(f"x1与x3的第{i}个数据不一致")
                return False
        print(f"x1与x3数据一致")
        for j in range(dim):
            if x2[j] != x3[j+dim]:
                print(f"x2与x3的第{j}个数据不一致")
                return False
        print(f"x2与x3数据一致")
        return True
    
    def compare_two_file(self, file1, file2, dtype:torch.dtype):
        np_dtype, _ = get_dtype(dtype)
        x1 = torch.tensor(np.fromfile(file1, dtype=np_dtype)).view(dtype=dtype)
        x2 = torch.tensor(np.fromfile(file2, dtype=np_dtype)).view(dtype=dtype)

        if torch.all(x1 == x2):
            print(f"两个文件的数据相同")
        else:
            print(f"两个文件的数据不相同，可以进行后续比较")

    def detect_matrix_input_output(self, dim_input, dim_weight,  file_input, file_weight, file_output, file_bias=None, dtype:torch.dtype=torch.bfloat16):
        np_dtype, data_size = get_dtype(dtype)
        input_HW = torch.tensor(np.fromfile(file_input, dtype=np_dtype)).view(dtype=dtype)
        weight_WH = torch.tensor(np.fromfile(file_weight, dtype=np_dtype)).view(dtype=dtype)
        output_HW = torch.tensor(np.fromfile(file_output, dtype=np_dtype)).view(dtype=dtype)
        if file_bias is not None:
            bias_HW = torch.tensor(np.fromfile(file_bias, dtype=np_dtype)).view(dtype=dtype)
            dim_bias = (dim_input[0],dim_weight[1])
            # bias_HW维度为weight[1], 需要广播为(dim_input[0],weight[1])
            bias_HW = bias_HW.expand(dim_input[0],dim_weight[1])
            # 1. 先还原input，weight，bias等成ddr，以及矩阵维度
            block_size = 512//data_size
            block_W = 32//data_size
            block_H = block_size//block_W
            input = HW_TO_ddr(input_HW, dim_input, block_size, block_W, block_H)
            weight = WH_TO_ddr(weight_WH, dim_weight, block_size, block_W, block_H)
            # bias = HW_TO_ddr(bias_HW, dim_bias, block_size, block_W, block_H)
            # 2. 计算ddr_output = input@weight + bias 的结果
            ddr_output = input @ weight + bias_HW
            # 3. 与output做对比
            res = slice_HW_compare(ddr_output, output_HW, block_size, block_W, block_H)
            if res:
                print(f"矩阵乘法运算结果正确")
            else:
                print(f"矩阵乘法运算结果错误")
                return False
        else:
            # 1. 先还原input，weight，bias等成ddr，以及矩阵维度
            block_size = 512//data_size
            block_W = 32//data_size
            block_H = block_size//block_W
            input = HW_TO_ddr(input_HW, dim_input, block_size, block_W, block_H)
            weight = WH_TO_ddr(weight_WH, dim_weight, block_size, block_W, block_H)
            # 2. 计算ddr_output = input@weight + bias 的结果
            ddr_output = input @ weight.t()
            # 3. 与output做对比
            res = slice_HW_compare(ddr_output, output_HW, block_size, block_W, block_H)
            if res:
                print(f"矩阵乘法运算结果正确")
            else:
                print(f"矩阵乘法运算结果错误")
                return False
