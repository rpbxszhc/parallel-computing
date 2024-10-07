import numpy as np
import random
def generate_random_input(M, N, P, sparsity=0.1):
    """
    生成随机输入数据，包括稠密矩阵D和稀疏矩阵S的非零元素。
    
    参数:
        M (int): 稠密矩阵的行数
        N (int): 稠密矩阵的列数和稀疏矩阵的行数
        P (int): 稀疏矩阵的列数
        sparsity (float): 稀疏矩阵的稀疏度，即非零元素占总元素的比例
    
    返回:
        str: 格式化的输入数据
    """
    # 生成稠密矩阵D
    D = np.random.randint(1, 10, size=(M, N))
    
    # 生成稀疏矩阵S
    num_elements = int(N * P * sparsity)
    row_indices = np.random.randint(0, N, size=num_elements)
    col_indices = np.random.randint(0, P, size=num_elements)
    values = np.random.randint(1, 10, size=num_elements)
    
    # 按列主序排序稀疏矩阵元素
    sorted_indices = np.lexsort((row_indices, col_indices))
    row_indices = row_indices[sorted_indices]
    col_indices = col_indices[sorted_indices]
    values = values[sorted_indices]
    
    # 生成输入数据字符串
    input_data = f"{M} {N} {P} {num_elements}\n"
    for row in D:
        input_data += " ".join(map(str, row)) + "\n"
    for r, c, v in zip(row_indices, col_indices, values):
        input_data += f"{r} {c} {v}\n"
    
    return input_data

def write_input_to_file(file_path, M, N, P, sparsity=0.1):
    """
    生成随机输入数据并将其写入文件。
    
    参数:
        file_path (str): 输出文件路径
        M (int): 稠密矩阵的行数
        N (int): 稠密矩阵的列数和稀疏矩阵的行数
        P (int): 稀疏矩阵的列数
        sparsity (float): 稀疏矩阵的稀疏度，即非零元素占总元素的比例
    """
    # 生成随机输入数据
    input_data = generate_random_input(M, N, P, sparsity)
    
    # 将数据写入文件
    with open(file_path, 'w') as file:
        file.write(input_data)

# 示例用法
# M, N, P = 40, 50, 30
M = random.randint(1000, 2000)
N = random.randint(1000, 2000)
P = random.randint(1000, 2000)
file_path = 'input'
write_input_to_file(file_path, M, N, P)
print(f"随机输入数据已写入文件 {file_path}")
