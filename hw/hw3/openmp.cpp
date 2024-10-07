// serial versions of the Buffon - Laplace Needle simulation
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <windows.h>
#include<fstream>
// drop n needles of length l onto grids, size of every cell of which is a * b

int thread = 2;

int BuffonLaplaceSimulation(double l, double a, double b, int n)
{
    int hits = 0;
    double x1, y1, x2, y2;
    double angle;
    omp_set_num_threads(thread);
    #pragma omp parallel for private(x1, y1, x2, y2, angle) reduction(+:hits) schedule(guided, 10)
    for (int i = 0; i < n; i++)
    {
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

int main(int argc, char *argv[])
{
    int needleNumber = 1000;
    double l = 1.0;
    double a = 1.0;
    double b = 1.0;
    std::ofstream out("openmp.txt");
    // thread = strtol(argv[1], NULL, 10);
    // (2l(a + b) - l^2) / (pi ab)
    for(int i = 0; i < 4; i++){
        needleNumber = 1000;
        for(int j = 0; j < 5; j++){
            LARGE_INTEGER t1,t2,tc;
            QueryPerformanceFrequency(&tc);
            QueryPerformanceCounter(&t1);
            double pi = (2 * l * (a + b) - pow(l, 2)) / (a * b) * (needleNumber) / (BuffonLaplaceSimulation(l, a, b, needleNumber));
            QueryPerformanceCounter(&t2);
            printf("thread num: %d\n", thread);
            printf("needle num: %d\n", needleNumber);
            printf("time: %lfs\n", double(t2.QuadPart-t1.QuadPart)/tc.QuadPart);
            printf("Pi = %lf\n\n", pi);
            if(needleNumber == 1000 || needleNumber == 10000000){
                out <<double(t2.QuadPart-t1.QuadPart)/tc.QuadPart << " ";
            }
            needleNumber *= 10;
        }
        thread += 2;
    }
    out <<'\n';
    out.close();
    return 0;
}