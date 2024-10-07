#include <fstream>
#include <iostream>
#include <stdio.h>
using namespace std;

#define PER 8

__global__ void spmm_kernel(const int *index, const int *indice, const int* data , const int* B, int* C, int NNZ, int M, int P, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * PER;
    int cnt = min(PER, M-row);
    if(cnt < 1) return;
    int sum[PER]={0};
    for(int i = index[col]; i < index[col+1]; i++){
        for(int j = 0; j < cnt; j++){
            sum[j] += data[i] * B[(row+j) * N + indice[i]];
        }
    }
    for(int j = 0; j < cnt; j++){
        C[(row+j) * P + col] = sum[j];
    }
    // if (row < M && col < P) {
    //     int sum = 0;
    //     for(int i = index[col]; i < index[col+1]; i++){
    //         sum += data[i] * B[row * N + indice[i]];
    //     }
    //     C[row * P + col] = sum;
    // }
}

int main(){
    int M,N,P,NNZ;
    cin>>M>>N>>P>>NNZ;
    int *data = new int[NNZ];
    int *index = new int[P+1];
    int *indice = new int[NNZ];
    int *B = new int[M*N];
    int *C = new int[M*P];
    index[0] = 0;
    memset(C, 0, sizeof(int) * M * P);
    for(int i = 0; i < M * N; i++){
        cin>>B[i];
    }
    int col = 0, tmp;
    for(int i = 0; i < NNZ; i++){
        cin>>indice[i]>>tmp>>data[i];
        if(tmp != col){
            for(int j = col+1; j <= tmp; j++){
                index[j] = i;
            }
            col = tmp;
        }
    }
    for(int j = col+1; j <= P; j++){
        index[j] = NNZ;
    }
    int *d_data, *d_index, *d_indice, *d_B, *d_C;
    cudaMalloc((void**)&d_data, sizeof(int) * NNZ);
    cudaMalloc((void**)&d_index, sizeof(int) * (P+1));
    cudaMalloc((void**)&d_indice, sizeof(int) * NNZ);
    cudaMalloc((void**)&d_B, sizeof(int) * M * N);
    cudaMalloc((void**)&d_C, sizeof(int) * M * P);
    cudaMemcpy(d_data, data, sizeof(int) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index, sizeof(int) * (P+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indice, indice, sizeof(int) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(int) * M * P, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y * PER - 1) / (blockSize.y * PER));
    spmm_kernel<<<gridSize, blockSize>>>(d_index, d_indice, d_data, d_B, d_C, NNZ, M, P, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeof(int) * M * P, cudaMemcpyDeviceToHost);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < P; j++){
            cout<<C[i * P + j]<<" ";
        }
        cout<<endl;
    }
    cudaFree(d_data);
    cudaFree(d_index);
    cudaFree(d_indice);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] data;
    delete[] index;
    delete[] indice;
    delete[] B;
    delete[] C;
    return 0;
}
