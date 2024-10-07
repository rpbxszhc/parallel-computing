import torch

# 输入处理
M, N, P, K = map(int, input().strip().split())

# 读取稠密矩阵D
D = torch.zeros((M, N), dtype=torch.int32)
for i in range(M):
    D[i] = torch.tensor(list(map(int, input().strip().split())), dtype=torch.int32)

# 读取稀疏矩阵S的非零元素
indices = []
values = []
for _ in range(K):
    r, c, v = map(int, input().strip().split())
    indices.append([r, c])
    values.append(int(v))

# 转置稀疏矩阵的坐标表示为 PyTorch 张量
indices = torch.tensor(indices, dtype=torch.int64).t()  # 转置以符合稀疏矩阵格式
values = torch.tensor(values, dtype=torch.int32)

# 创建稀疏矩阵S
S = torch.sparse.IntTensor(indices, values, torch.Size([N, P]))

# 将稠密矩阵和稀疏矩阵复制到 GPU
D = D.cuda()
S = S.cuda()

# 计算稠密矩阵与稀疏矩阵的乘积
result = torch.matmul(D.to(dtype=torch.float32), S.to(dtype=torch.float32)).to(dtype=torch.int32)

# 将结果复制回 CPU 并输出
result = result.cpu()

for i in range(M):
    print(' '.join(map(str, result[i].tolist())))
