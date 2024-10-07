#include <iostream>
#include <fstream>

using namespace std;

#define STEP 8192
#define SIZE(A, B) ((A+B-1)/B)

__global__ void spmm_kernel(const int *index, const int *indice, const int* data , const int* B, int* C, int NNZ, int M, int P, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (row < M && col < P && idx < SIZE(N,STEP)) {
        int sum = 0;
        for(int i = index[col]+idx; i < index[col+1]; i+=SIZE(N,STEP)){
            sum += data[i] * B[row * N + indice[i]];
        }
        C[idx * M * P + row * P + col] = sum;
    }
}

__global__ void merge(int * C, int M, int N, int P){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < M && col < P){
        int sum = 0;
        for(int i = 0; i < SIZE(N,STEP); i++){
            sum += C[i * M * P + row * P + col];
        }
        C[row * P + col] = sum;
    }
}

int main(){
    int M,N,P,NNZ;
    cin>>M>>N>>P>>NNZ;
    int *data = new int[NNZ];
    int *index = new int[P+1];
    int *indice = new int[NNZ];
    int *B = new int[M*N];
    int *C = new int[M*P* SIZE(N,STEP)];
    index[0] = 0;
    memset(C, 0, sizeof(int) * M * P * SIZE(N,STEP));
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
    cudaMalloc((void**)&d_C, sizeof(int) * M * P * SIZE(N,STEP));
    cudaMemcpy(d_data, data, sizeof(int) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index, sizeof(int) * (P+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indice, indice, sizeof(int) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(int) * M * P * SIZE(N,STEP), cudaMemcpyHostToDevice);
    dim3 blockSize(16,16,1);
    dim3 gridSize(SIZE(P, blockSize.x), SIZE(M, blockSize.y), SIZE(N,STEP));
    spmm_kernel<<<gridSize, blockSize>>>(d_index, d_indice, d_data, d_B, d_C, NNZ, M, P, N);
    cudaDeviceSynchronize();
    merge<<<dim3(SIZE(P,16), SIZE(M,16)), dim3(16,16)>>>(d_C, M, N, P);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeof(int) * M * P * SIZE(N, STEP), cudaMemcpyDeviceToHost);
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
