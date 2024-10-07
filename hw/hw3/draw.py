import matplotlib.pyplot as plt
prog_num = [2,4,6,8]

with open('mpi.txt', 'r') as f:
    string = f.readline()
    mpi = [float(i) for i in string.split()]
    mpi10_3 = mpi[::2]
    mpi10_7 = mpi[1::2]
    
with open('openmp.txt', 'r') as f:
    string = f.readline()
    openmp = [float(i) for i in string.split()]
    openmp10_3 = openmp[::2]
    openmp10_7 = openmp[1::2]


plt.subplot(121)
plt.title('needle num = 10^3')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('处理器数')  # x轴标题
plt.ylabel('时间')  # y轴标题

plt.plot(prog_num, mpi10_3, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
# plt.plot(prog_num, mpi10_7, marker='o', markersize=3)
plt.plot(prog_num, openmp10_3, marker='o', markersize=3)
# plt.plot(prog_num, openmp10_7, marker='o', markersize=3)

plt.legend(['mpi', 'openmp'])  # 设置折线名称


plt.subplot(122)
plt.title('needle num = 10^7')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('处理器数')  # x轴标题
plt.ylabel('时间')  # y轴标题

# plt.plot(prog_num, mpi10_3, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(prog_num, mpi10_7, marker='o', markersize=3)
# plt.plot(prog_num, openmp10_3, marker='o', markersize=3)
plt.plot(prog_num, openmp10_7, marker='o', markersize=3)

plt.legend(['mpi', 'openmp'])  # 设置折线名称

plt.show()  # 显示折线图
