//kmeans算法mpi实现
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <iomanip>

using namespace std;

#define ITER 100

int N, D, K;

int main(int argc,char *argv[]){
    int rank,size;
    ifstream infile("case.in");
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(!rank) {
        // cin>>N>>D>>K;
        infile >> N >> D >> K;
    }
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&D,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&K,1,MPI_INT,0,MPI_COMM_WORLD);
    float data[N][D];
    float cluster_center[K][D];
    float local_cluster_center[K][D];//每次聚类得到的新聚类中心
    int num;//每个进程分配的数据量
    if(rank==0){
        for(int i=0;i<N;i++){
            for(int j=0;j<D;j++){
                // cin >> data[i][j];
                infile >> data[i][j];
            }
        }
        num = N / size;
        int remain = N % size+num;
        for(int i=1;i<size - 1;i++){
            MPI_Send(&num,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
        MPI_Send(&remain, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD);
    }   
    else{
        MPI_Recv(&num,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);  //同步一下
    if(rank==0){
        //分发数据,以字节的类型发送，一次send将所有数据发送给接收方
        for(int i = 1; i < size - 1; i++){
            int start=num*i;
            MPI_Send((void*)(data+start),num*D,MPI_FLOAT,i,99,MPI_COMM_WORLD);
        }
        MPI_Send((void*)(data+num*(size-1)),(N%size+num)*D,MPI_FLOAT,size-1,99,MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(data,num * D,MPI_FLOAT,0,99,MPI_COMM_WORLD,&status);
    }
    // MPI_Barrier(MPI_COMM_WORLD);  //同步一下

    //进程0产生随机中心点
    if(rank==0){
        srand((unsigned int)(time(NULL)));  
        int per = N / K;
        for(int i = 0; i < K; i++){
            int idx = rand() % per;
            for(int j = 0; j < D; j++){
                cluster_center[i][j] = data[idx+i*per][j];
            }
        }
    }
    MPI_Bcast(cluster_center,K*D,MPI_FLOAT,0,MPI_COMM_WORLD);

    int local_cnt[K],total_cnt[K];
    for(int round=0;round<ITER;round++){
        memset(local_cluster_center,0,sizeof(local_cluster_center));
        memset(local_cnt,0,sizeof(local_cnt));
        for(int i = 0; i < num; i++){
            int min_idx = 0;
            float min_dis = 1e9;
            for(int j = 0; j < K; j++){
                float dis = 0;
                for(int k = 0; k < D; k++){
                    dis += (data[i][k] - cluster_center[j][k]) * (data[i][k] - cluster_center[j][k]);
                }
                if(dis < min_dis){
                    min_dis = dis;
                    min_idx = j;
                }
            }
            local_cnt[min_idx]++;
            for(int j = 0; j < D; j++){
                local_cluster_center[min_idx][j] += data[i][j];
            }
        }
        memset(cluster_center,0,sizeof(cluster_center));
        memset(total_cnt,0,sizeof(total_cnt));
        MPI_Reduce(local_cluster_center,cluster_center,K*D,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(local_cnt,total_cnt,K,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        // MPI_Barrier(MPI_COMM_WORLD);

        if(!rank){
            for(int i = 0; i < K; i++){
                for(int j = 0; j < D; j++){
                    if(total_cnt[i] != 0){
                        cluster_center[i][j] /= total_cnt[i];
                    }
                }
            }
        }
        MPI_Bcast(cluster_center,K*D,MPI_FLOAT,0,MPI_COMM_WORLD);
    }
    float total_dis = 0;
    for(int i = 0; i < num; i++){
        float min_dis = 1e9;
        for(int j = 0; j < K; j++){
            float dis = 0;
            for(int k = 0; k < D; k++){
                dis += (data[i][k] - cluster_center[j][k]) * (data[i][k] - cluster_center[j][k]);
            }
            if(sqrt(dis) < min_dis){
                min_dis = sqrt(dis);
            }
        }
        total_dis += min_dis;
    }
    float dis_final = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&total_dis, &dis_final, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(!rank){
        std::cout << std::fixed << std::setprecision(2) << dis_final << std::endl;
    }
    MPI_Finalize();
    return 0;
}