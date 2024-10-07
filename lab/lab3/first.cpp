//kmeans算法mpi实现
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <math.h>

using namespace std;

const int K=7; //聚类的数目
const int D=16;//数据的维数
const int epoch=100;//迭代轮数

unordered_map<int,string> idx2name;

//自定义结构体
struct animal{
    int name;//在idx2name中的索引
    int type;
    int characters[D];
};

//从zoo.data中读取数据
int loadData(string filename,animal* &data){
    ifstream infile;
    infile.open(filename);
    if(!infile) cout<<"failed to open file "+filename+" !\n";
    string str;
    int dataNum=0;
    vector<animal> tmp;
    while(infile>>str){
        animal curline;
        int i=0;
        //保存名字
        string name="";
        while(str[i]!=',')name+=str[i++];
        i++;
        //确定名字到整数索引的映射
        idx2name[dataNum]=name;
        curline.name=dataNum++;
        //保存特征
        for(int j=0;j<D;j++){
            curline.characters[j]=str[i]-'0';
            i+=2;
        }
        //保存所属类型
        int type=str[i]-'0';
        curline.type=type;
        tmp.push_back(curline);
    }
    data=new animal[dataNum];
    for(int i=0;i<dataNum;i++){
        data[i]=tmp[i];
    }
    return dataNum;
}

//求欧式距离
double distance(int charc[],double center_charc[]){
    double dis=0.0;
    for(int i=0;i<D;i++){
        dis+=(charc[i]*1.0-center_charc[i])*(charc[i]*1.0-center_charc[i]);
    }
    return sqrt(dis);
}

//聚类
void cluster(animal* &data,int dataSize,double data_center[][D],double new_data_center[][D],int cnt[]){
    for(int i=0;i<dataSize;i++){
        double min_dis=10000.0;
        int clusterId=-1;
        for(int j=0;j<K;j++){
            double cur_dis=distance(data[i].characters,data_center[j]);
            if(cur_dis<min_dis){
                min_dis=cur_dis;
                clusterId=j;
            }
        }
        //便于后续计算新的聚类中心
        for(int j=0;j<D;j++){
            new_data_center[clusterId][j]+=data[i].characters[j];
        }
        cnt[clusterId]++;//每一类包含的个数
        data[i].type=clusterId;//保存新的所属的类别
    }
}

int main(int argc,char *argv[]){
    int rank,size;
    int dataNum;//每个进程处理的数据数
    animal* data;//保存数据
    double cluster_center[K][D];//数据聚类中心点
    memset(cluster_center,0,sizeof(cluster_center));
    double local_cluster_center[K][D];//每次聚类得到的新聚类中心
    MPI_Status status;
    clock_t startTime,endTime;
    startTime = clock();
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    //进程0读取数据，同时告知每个进程它需要处理的数据量
    if(rank==0){
        // dataNum=loadData("case.in",data);
        // for(int i=1;i<size;i++){
        //     int nums=dataNum/(size-1);
        //     int start=(i-1)*nums;
        //     int end=i*nums;
        //     if(i==size-1)end=dataNum;
        //     int sendNum=end-start;
        //     MPI_Send(&sendNum,1,MPI_INT,i,99,MPI_COMM_WORLD);
        // }

        // cin >> N >> M >> S;
	    // tent.resize(N, INT_MAX);
        // B.resize(Bsize);
        // Graph.resize(N);
        // for(int i=0; i<M; i++){
        //     cin >> a >> b >> c;
        //     Graph[a].push_back({b, c});
	    // 	Graph[b].push_back({a, c});
        // }
	    ifstream infile("./case.in");
        infile >> dataNum >> D >> K;
        for(int i=0; i<M; i++){
            infile >> a >> b >> c;
        }
        
    }
    else{
        MPI_Recv(&dataNum,1,MPI_INT,0,99,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);  //同步一下

    if(rank==0){
        //分发数据,以字节的类型发送，一次send将所有数据发送给接收方
        for(int i=1;i<size;i++){
            int nums=dataNum/(size-1);
            int start=(i-1)*nums;
            int end=i*nums;
            if(i==size-1)end=dataNum;
            MPI_Send((void*)(data+start),sizeof(animal)*(end-start),MPI_BYTE,i,99,MPI_COMM_WORLD);
        }
        
    }
    else{
        data=new animal[dataNum];
        MPI_Recv(data,sizeof(animal)*dataNum,MPI_BYTE,0,99,MPI_COMM_WORLD,&status);
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  //同步一下
    
    //进程0产生随机中心点
    if(rank==0){
        srand((unsigned int)(time(NULL)));  
        
        unordered_set<int> vis;
        int i=0;
        while(i<K){
            int idx=rand()%dataNum;
            //该数据没被选择过
            if(vis.count(idx)==0){
                for(int j=0;j<D;j++)cluster_center[i][j]=data[idx].characters[j];
                vis.insert(idx);
                i++;
            }
        }
    }
    //广播数据中心
    MPI_Bcast(cluster_center,K*D,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    //开始做聚类
    int local_cnt[K],total_cnt[K];
    for(int round=0;round<epoch;round++){
        
        memset(local_cluster_center,0,sizeof(local_cluster_center));
        memset(local_cnt,0,sizeof(local_cnt));
        if(rank){
            cluster(data,dataNum,cluster_center,local_cluster_center,local_cnt);
        }
        memset(cluster_center,0,sizeof(cluster_center));
        memset(total_cnt,0,sizeof(total_cnt));

        //将local_cluster_center规约到进程0以便计算新的聚类中心
        
        MPI_Reduce(local_cluster_center,cluster_center,D*K,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        
        MPI_Reduce(local_cnt,total_cnt,K,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);  //同步一下

        if(rank==0){
            //计算新的聚类中心
            for(int i=0;i<K;i++){
                
                for(int j=0;j<D;j++){   
                    if(total_cnt[i]!=0)
                    cluster_center[i][j]/=total_cnt[i];
                }
                
            }
        }
        //广播新的中心
        MPI_Bcast(cluster_center,K*D,MPI_DOUBLE,0,MPI_COMM_WORLD);
       
    }
    
    //收集数据，打印结果
    if(rank){
        int buf[dataNum*2];
        for(int i=0;i<dataNum;i++){
            buf[i*2]=data[i].name;
            buf[i*2+1]=data[i].type;
        }
        MPI_Send(buf,dataNum*2,MPI_INT,0,99,MPI_COMM_WORLD);
    }else{
        int buf[dataNum*2];
        for(int i=1;i<size;i++){
            int nums=dataNum/(size-1);
            int start=(i-1)*nums;
            int end=i*nums;
            if(i==size-1)end=dataNum;
            int sendNum=end-start;
            MPI_Recv(&buf[start*2],sendNum*2,MPI_INT,i,99,MPI_COMM_WORLD,&status);
        }
        
        vector<string> clusters[K];
        for(int i=0;i<dataNum;i++){
            clusters[buf[i*2+1]].push_back(idx2name[buf[i*2]]);
        }
        string filename="clusters-mpi.txt";
        ofstream out(filename);
        for(int i=0;i<K;i++){
            out<<"cluster-"<<i<<":"<<endl;
            int cnts=1;
            for(string name:clusters[i]){
                if(cnts%6==0)
                    out<<name<<endl;
                else out<<name<<" ,";
                cnts++;
            }
            out<<endl<<endl;
        }
        out.close();
    }
    delete []data;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    endTime = clock();
    cout <<rank<< " : The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}
