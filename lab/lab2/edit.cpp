#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <climits>
#include <fstream>
#include <algorithm>
#include <cmath>


using namespace std;

int N, M, S;
typedef std::vector<int> bucketType;
typedef std::vector<bucketType> bucketArray;
struct edge
{
    int dest;
    int weight;
};


int delta = 1000;
int MAX_WEIGHT = INT_MAX;
bucketArray buckets;
omp_lock_t* locks;
std::vector<int> dists;
std::vector< std::vector<edge>> Graph;
int NUM_THREADS = 8;
int threhold = 64;
int min_bucket_index = INT_MAX;
bool flag = false;

void delta_stepping(std::vector< std::vector<edge>> G, int source);
void init_buckets(int source);
void relax(int w, int d);
void relax_parallel(int source, bucketArray& local_buckets);
void bucket_fusion_parallel(bucketArray& local_buckets);
void copy_to_shared_buckets(bucketArray& local_buckets);

int main() {
    omp_set_num_threads(NUM_THREADS);
    int temp;
    struct edge tmpedge;
    // ifstream infile("./case.in");
    // infile >> N;
    // infile >> M;
    // infile >> S;
    // Graph.resize(N);
    // for(int i=0; i<M; i++){
    //     infile >> temp;
    //     infile >> tmpedge.dest;
    //     infile >> tmpedge.weight;
    //     Graph[temp].push_back(tmpedge);
    //     int tmp = tmpedge.dest;
    //     tmpedge.dest = temp;
    //     Graph[tmp].push_back(tmpedge);
    // }
    // infile.close();
    cin >> N >> M >> S;
    Graph.resize(N);
    for(int i=0; i<M; i++){
        cin >> temp;
        cin >> tmpedge.dest;
        cin >> tmpedge.weight;
        Graph[temp].push_back(tmpedge);
        int tmp = tmpedge.dest;
        tmpedge.dest = temp;
        Graph[tmp].push_back(tmpedge);
    }
    delta_stepping(Graph, S);
    // ifstream outfile("./case.out");
    for(int i = 0; i < N; i++){
        // outfile >> temp;
        if(dists[i]!=INT_MAX){
            cout << dists[i] << " ";
        }
        else{
            cout << "INF ";
        }
    }
    return 0;
}


void delta_stepping(std::vector< std::vector<edge>> G, int source) {
    init_buckets(source);
    /* shared variables */
    for(int i=0; i<buckets.size(); i++){
        if(!buckets[i].empty()){
            min_bucket_index = i;
            break;
        }
    }
    bucketType *min_bucket = &buckets.at(min_bucket_index);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        bucketArray local_buckets;
        while(min_bucket_index != INT_MAX){
            #pragma omp for nowait schedule(dynamic, 8)
            for(int i=0; i<min_bucket->size(); i++){
                int source = min_bucket->at(i);
                relax_parallel(source, local_buckets);
            }

            bucket_fusion_parallel(local_buckets);
            #pragma omp barrier
            {
                copy_to_shared_buckets(local_buckets);
            }
            local_buckets.clear();
            #pragma omp barrier
            #pragma omp single nowait
            {
                if(!flag){
                    min_bucket->clear();
                    min_bucket_index = INT_MAX;
                    for(int i=0; i<buckets.size(); i++){
                        if(!buckets.at(i).empty()){
                            min_bucket_index = i;
                            break;
                        }
                    }
                    if (min_bucket_index != INT_MAX){
                        min_bucket = &buckets.at(min_bucket_index);
                    }
                }
                flag = false;
            }
            #pragma omp barrier
        }
    }
    for(int i=0; i<sizeof(locks)/sizeof(locks[0]); i++){
        omp_destroy_lock(&locks[i]);
    }
}

void init_buckets(int source){
    int num_buckets = ceil(MAX_WEIGHT / delta) + 1;
    buckets.resize(num_buckets);
    dists.resize(N, INT_MAX);
    relax(source, 0);
    locks = new omp_lock_t[num_buckets];
    for (int i=0; i<num_buckets; i++) {
        omp_init_lock(&locks[i]);
    }
}

void relax(int w, int d){
    if(d < dists[w]){
        buckets[d/delta].push_back(w);
        dists[w] = d;
    }
}

void relax_parallel(int source, bucketArray& local_buckets){
    for(int j=0; j<Graph[source].size(); j++){
        int weight;
        #pragma omp critical
        weight = Graph[source][j].weight;
        int &old_dist = dists[Graph[source][j].dest];
        int new_dist = dists[source] + weight;
        if (new_dist < old_dist) {
            // #pragma omp atomic write
            dists[Graph[source][j].dest] = new_dist;
            auto bucket_index = new_dist / delta;
            if (bucket_index >= local_buckets.size()) {
                local_buckets.resize(bucket_index + 1);
            }
            if(bucket_index <= min_bucket_index){
                #pragma omp atomic write
                flag = true;
            }
            local_buckets[bucket_index].push_back(Graph[source][j].dest);
        }
    }
}

void bucket_fusion_parallel(bucketArray& local_buckets){
    if (!local_buckets.empty()){
        int i = 0;
        while(local_buckets[i].empty()){
            i++;
        }
        while(!local_buckets[i].empty() && local_buckets[i].size() < threhold){
            auto local_bucket_copy = local_buckets[i];
            local_buckets[i].resize(0);
            for (auto source: local_bucket_copy){
                relax_parallel(source, local_buckets);
            }
        }
    }
}

void copy_to_shared_buckets(bucketArray& local_buckets){
    for (int i=0; i<local_buckets.size(); i++){
        if(!local_buckets[i].empty()){
            omp_set_lock(&locks[i]);
            bucketType &local_ref_bucket = local_buckets[i];
            if(buckets.size() <= i){
                buckets.resize(i+1);
            }
            buckets[i].insert(buckets[i].end(), local_ref_bucket.begin(), local_ref_bucket.end());
            omp_unset_lock(&locks[i]);
        }
    }
}