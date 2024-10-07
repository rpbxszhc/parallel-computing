import subprocess
import time

start_time = time.time()

# 替换为你的可执行文件路径
cmd_command = "mpiexec -n 8 self.exe"  # 例如这里是一个简单的dir命令

# 运行可执行文件
for _ in range(10):
    result = subprocess.run(cmd_command, shell=True, capture_output=True, text=True)
    print(result.stdout)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")