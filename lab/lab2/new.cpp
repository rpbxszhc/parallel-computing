#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <climits>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <set>

int N, M, S;
#define NUM_THREADS 8
#define Bsize 1000000
using namespace std;


int delta = 1000;
int max_bucket, min_bucket=0;
std::vector<int> tent;
vector<std::vector<int>> B;
struct req{
    int w;
    int d;
};
std::vector<req> Req;

void delta_stepping(std::vector< std::vector<req> > &G, int source);
bool b_is_empty();
void relax(int w, int d);



int main(int argc, char *argv[]) {
    omp_set_num_threads(NUM_THREADS);
    std::vector< std::vector<req>> Graph;
	int a, b, c;
    cin >> N >> M >> S;
	tent.resize(N, INT_MAX);
    B.resize(Bsize);
    Graph.resize(N);
    for(int i=0; i<M; i++){
        cin >> a >> b >> c;
        Graph[a].push_back({b, c});
		Graph[b].push_back({a, c});
    }
	// ifstream infile("./case.in");
    // infile >> N >> M >> S;
    // tent.resize(N, INT_MAX);
    // B.resize(N);
    // Graph.resize(N);
    // for(int i=0; i<M; i++){
    //     infile >> a >> b >> c;
    //     Graph[a].push_back({b, c});
	// 	Graph[b].push_back({a, c});
    // }
    // infile.close();

    delta_stepping(Graph, S);

    // int temp;
    // ifstream outfile("./case.out");
    for(int i = 0; i < N; i++){
        // outfile >> temp;
        if(tent[i]!=INT_MAX){
            cout << tent[i] << " ";
        }
        else{
            cout << "INF ";
        }
    }
    return 0;
}

bool b_is_empty(){
    for(int i=min_bucket; i< max_bucket; i++){
        if(!B[i].empty()){
            return false;
        }
    }
    return true;
}

void relax(int w, int d){
    if(d < tent[w]){
        if(tent[w] != INT_MAX){
            auto it = find(B[tent[w]/delta].begin(), B[tent[w]/delta].end(), w);
            if(it!=B[tent[w]/delta].end()){
                B[tent[w]/delta].erase(it);
            }
        }
        B[d/delta].push_back(w);
        if(d/delta >= max_bucket){
            max_bucket = d / delta + 1;
        }
        tent[w] = d;
    }
}


void delta_stepping(std::vector< std::vector<req> > &G, int source) {
    std::vector<int> S;
    std::vector< std::vector<int> > light(N, std::vector<int>(0));
    std::vector< std::vector<int> > heavy(N, std::vector<int>(0));
    int i;
    #pragma omp parallel for
    for(i=0; i<N; i++){
        for(int j=0; j<G[i].size(); j++){
            if(G[i][j].d > delta){
                heavy[i].push_back(j);
            } 
            else{
                light[i].push_back(j);
            }
        }
    }
    max_bucket = 0;
    relax(source, 0);
    i = 0;
    while(!b_is_empty()){
        S.clear();
        while(!B[i].empty()){
            Req.clear();
            #pragma omp for
            for(int j=0; j<B[i].size(); j++){
                int v = B[i][j];
                for(int k=0; k<light[v].size(); k++){
                    int w = G[v][light[v][k]].w;
                    req r;
                    r.w = w;
                    r.d = tent[v] + G[v][light[v][k]].d;
                    Req.push_back(r);
                }
            }
            S.insert(S.end(), B[i].begin(), B[i].end());
            B[i].clear();
            #pragma omp for
            for(int j=0; j<Req.size(); j++){
                relax(Req[j].w, Req[j].d);
            }
        }
        min_bucket = i + 1;
        Req.clear();
        #pragma omp for
        for(int j=0; j<S.size(); j++){
            int v=S[j];
            for(int k=0; k<heavy[v].size(); k++){
                int w = G[v][heavy[v][k]].w;
                req r;
                r.w = w;
                r.d = tent[v] + G[v][heavy[v][k]].d;
                Req.push_back(r);
            }
        }
        #pragma omp for
        for(int j=0; j<Req.size(); j++){
            relax(Req[j].w, Req[j].d);
        }
        i++;
    }
}