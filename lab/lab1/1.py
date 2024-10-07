import numpy as np

def generate_upper_triangular(n):
    # 生成随机的上三角方阵
    upper_triangular = np.triu(np.random.randint(1, 10, size=(n, n)))
    return upper_triangular

def generate_lower_triangular(n):
    # 生成随机的下三角方阵
    lower_triangular = np.tril(np.random.randint(1, 10, size=(n, n)))
    return lower_triangular

def matrix_multiply(lower_triangular, upper_triangular):
    # 计算两个矩阵的乘积
    result = np.dot(lower_triangular, upper_triangular)
    return result

# 生成一个3x3的上三角方阵和下三角方阵
n = 3
upper_triangular = generate_upper_triangular(n)
lower_triangular = generate_lower_triangular(n)

print("Upper Triangular Matrix:")
print(upper_triangular)
print("\nLower Triangular Matrix:")
print(lower_triangular)

# 计算两个矩阵的乘积
result = matrix_multiply(lower_triangular, upper_triangular)
print("\nMatrix Product:")
print(result)
