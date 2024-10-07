// serial versions of the Buffon - Laplace Needle simulation
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
// drop n needles of length l onto grids, size of every cell of which is a * b
int BuffonLaplaceSimulation(double l, double a, double b, int n)
{
    int hits = 0;
    double x1, y1, x2, y2;
    double angle;
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

int main()
{
    int needleNumber = 1000000;
    double l = 1.0;
    double a = 1.0;
    double b = 1.0;
    // (2l(a + b) - l^2) / (pi ab)
    LARGE_INTEGER t1,t2,tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);
    double pi = (2 * l * (a + b) - pow(l, 2)) / (a * b) * (needleNumber) / (BuffonLaplaceSimulation(l, a, b, needleNumber));
    QueryPerformanceCounter(&t2);
    printf("needle number: %d\n", needleNumber);
    printf("time: %lfs\n", double(t2.QuadPart-t1.QuadPart)/tc.QuadPart);
    printf("Pi = %lf\n\n", pi);
    return 0;
}