@echo off
setlocal
setlocal enabledelayedexpansion

REM 设置要运行的次数
set "iterations=4"
set "pro_cnt=2"

echo Running seral
g++ -g -o seral seral.cpp
seral.exe

echo --------------------------------------------
REM 循环运行
echo Running mpi:
g++ -o mpi mpi.cpp -fopenmp -l msmpi -L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
for /l %%i in (1, 1, %iterations%) do (
    mpiexec -n !pro_cnt! mpi.exe
    timeout /t 1 /nobreak >nul
    set /a "pro_cnt=pro_cnt+2"
)

echo --------------------------------------------

echo Running openmp
g++ -g -fopenmp -o openmp openmp.cpp
openmp.exe


echo completed.

endlocal
