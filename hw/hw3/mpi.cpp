#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include<fstream>
#include<iostream>

using namespace std;

// 模拟在 a * b 网格上投掷长度为 l 的针，返回交叉线的针数
int BuffonLaplaceSimulation(double l, double a, double b, int n) {
    int hits = 0;
    double x1, y1, x2, y2, angle;

    for (int i = 0; i < n; i++) {
        x1 = a * (double)rand() / (double)RAND_MAX;
        y1 = b * (double)rand() / (double)RAND_MAX;
        angle =  (double)rand();
        x2 = x1 + l * cos(angle);
        y2 = y1 + l * sin(angle);

        if (x2 <= 0 || x2 >= a || y2 <= 0 || y2 >= b)
            hits++;
    }
    return hits;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double l = 1.0; // 针的长度
    double a = 1.0; // 网格宽度
    double b = 1.0; // 网格高度
    std::ofstream out("mpi.txt", ios::out|ios::app);

    int needleNumber = 1000; // 总的针数

    for(int i = 0; i < 5; i++){

        int local_n = needleNumber / size; // 每个进程的针数
        int local_hits, total_hits = 0;
        double pi;

        LARGE_INTEGER t1,t2,tc;
        QueryPerformanceFrequency(&tc);
        QueryPerformanceCounter(&t1);
        local_hits = BuffonLaplaceSimulation(l, a, b, local_n);
        MPI_Reduce(&local_hits, &total_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // 计算并打印结果
            pi = (2 * l * (a + b) - pow(l, 2)) / (a * b) * (needleNumber) / total_hits;
            double end = MPI_Wtime();
            printf("size: %d\n", size);
            printf("needleNumber: %d\n", needleNumber);
            QueryPerformanceCounter(&t2);
            printf("time: %lfs\n", double(t2.QuadPart-t1.QuadPart)/tc.QuadPart);
            printf("Pi = %lf\n\n", pi);
            if(needleNumber == 1000 || needleNumber == 10000000){
                out << double(t2.QuadPart-t1.QuadPart)/tc.QuadPart << " ";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        needleNumber *= 10;
    }
    out.close();
    MPI_Finalize();
    return 0;
}