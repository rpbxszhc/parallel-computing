#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int x;
    int y;
    int val;
} sparse;

__global__ void MatMulti(int* D, int* Srow, int* Scol, int* Sval, int* Q, int* tag, int M, int N, int P, int* flag) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.x * blockDim.x + threadIdx.x);
    if(row < M && col < P) {
        int sum = 0;
        for(int k = 1; k <= tag[col*(N+1)]; k++) {
            sum += D[row * N + Srow[tag[col*(N+1)+k]]] * Sval[tag[col*(N+1)+k]];
        }
        Q[row * P + col] = sum;
    }
    *flag +=1;
    
}

int main() {
    int M, N, P, K;
    scanf("%d %d %d %d", &M, &N, &P, &K);

    int* D = (int*)malloc(sizeof(int) * M * N);
    int* tag = (int*)malloc(sizeof(int) * P * (N+1));
    for(int j = 0; j < M * N; j++) {
        scanf("%d", &D[j]);
    }
    for(int i = 0; i < P; i++)  tag[i*(N+1)] = 0;

    int* Srow = (int*)malloc(sizeof(int) * K);
    int* Scol = (int*)malloc(sizeof(int) * K);
    int* Sval = (int*)malloc(sizeof(int) * K);
    for(int i = 0; i < K; i++) {
        scanf("%d %d %d", &Srow[i], &Scol[i], &Sval[i]);
        tag[Scol[i]*(N+1) + tag[Scol[i]*(N+1)] + 1] = i;
        tag[Scol[i]*(N+1)] += 1;
    }

    int* Q = (int*)malloc(sizeof(int) * M * P);
    for(int i = 0; i < M * P; i++) {
        Q[i] = 0;
    }

    int* flag = (int*)malloc(sizeof(int));
    flag = 0;

    int *d_D, *d_Q, *d_tag;
    int *d_Srow, *d_Scol, *d_Sval;
    int* d_flag;

    cudaMalloc((void**)&d_D, sizeof(int) * M * N);
    cudaMalloc((void**)&d_Srow, sizeof(int) * K);
    cudaMalloc((void**)&d_Scol, sizeof(int) * K);
    cudaMalloc((void**)&d_Sval, sizeof(int) * K);
    cudaMalloc((void**)&d_Q, sizeof(int) * M * P);
    cudaMalloc((void**)&d_tag, sizeof(int) * P * (N+1));
    cudaMalloc((void**)&d_flag, sizeof(int));

    cudaMemcpy(d_D, D, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Srow, Srow, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Scol, Scol, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sval, Sval, sizeof(int) * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, sizeof(int) * M * P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tag, tag, sizeof(int) * P * (N+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, flag, sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((P-1)/16+1, (M-1)/16+1);

    MatMulti<<<dimGrid, dimBlock>>>(d_D, d_Srow, d_Scol, d_Sval, d_Q, d_tag, M, N, P, d_flag);

    cudaDeviceSynchronize();

    cudaMemcpy(Q, d_Q, sizeof(int) * M * P, cudaMemcpyDeviceToHost);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < P; j++) {
            printf("%d ", Q[i * P + j]);
        }
        printf("\n");
    }

    free(D);
    free(Srow);
    free(Scol);
    free(Sval);
    free(Q);
    free(tag);
    free(flag);
    cudaFree(d_D);
    cudaFree(d_Srow);
    cudaFree(d_Scol);
    cudaFree(d_Sval);
    cudaFree(d_Q);
    cudaFree(d_tag);

    return 0;
}