#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <climits>
#include <fstream>
#include <time.h>
#include <algorithm>

int N, M, S;
#define NUM_THREADS 8
#define Bsize 1000000
using namespace std;


int delta = 1000000000/Bsize;
int max_bucket;
std::vector<int> tent;
vector<std::vector<int>> B;
struct req{
    int w;
    int d;
};
std::vector<req> Req;
omp_lock_t* locks;

void delta_stepping(std::vector< std::vector<req> > &G, int source);
bool b_is_empty();
void relax(int w, int d);

int getidx(vector<req> &v, int w){
    for(int i=0; i<v.size(); i++){
        if(v[i].w == w){
            return i;
        }
    }
    cout << "Error: no such edge" << endl;
    exit(1);
}


int main(int argc, char *argv[]) {
    omp_set_num_threads(NUM_THREADS);
    std::vector< std::vector<req>> Graph;
	int a, b, c;
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
    infile >> N >> M >> S;
    tent.resize(N, INT_MAX);
    B.resize(N);
    Graph.resize(N);
    for(int i=0; i<M; i++){
        infile >> a >> b >> c;
        Graph[a].push_back({b, c});
		Graph[b].push_back({a, c});
    }
    infile.close();

    delta_stepping(Graph, S);

    int temp;
    ifstream outfile("./case.out");
    for(int i = 0; i < N; i++){
        outfile >> temp;
        if(tent[i]!=temp){
            cout << tent[i] << " ";
        }
        // else{
        //     cout << "INF ";
        // }
    }
    return 0;
}

bool b_is_empty(){
    for(int i=0; i< max_bucket; i++){
        if(!B[i].empty()){
            return false;
        }
    }
    return true;
}

void relax(int w, int d){
    if(d < tent[w]){
        if(tent[w] != INT_MAX){
            omp_set_lock(&locks[tent[w]/delta]);
            auto it = find(B[tent[w]/delta].begin(), B[tent[w]/delta].end(), w);
            if(it!=B[tent[w]/delta].end()){
                B[tent[w]/delta].erase(it);
            }
            omp_unset_lock(&locks[tent[w]/delta]);
        }
        omp_set_lock(&locks[d/delta]);
        B[d/delta].push_back(w);
        omp_unset_lock(&locks[d/delta]);
        if(d/delta >= max_bucket){
            #pragma omp critical
            max_bucket = d / delta + 1;
        }
        #pragma omp critical
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
                heavy[i].push_back(G[i][j].w);
                // heavy[i].push_back(j);
            } 
            else{
                light[i].push_back(G[i][j].w);
                // light[i].push_back(j);
            }
        }
    }
    locks = new omp_lock_t[Bsize];
    #pragma omp for
    for (int i=0; i<Bsize; i++) {
        omp_init_lock(&locks[i]);
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
                    // int w = G[v][k].w;
                    // req r;
                    // r.w = w;
                    // r.d = tent[v] + G[v][k].d;
                    int w = light[v][k];
                    req r;
                    r.w = w;
                    r.d = tent[v] + G[v][getidx(G[v], w)].d;
                    #pragma omp critical
                    {
                        Req.push_back(r);
                    }
                }
                #pragma omp critical
                {
                    S.push_back(v);
                }
            }
            B[i].clear();
            #pragma omp for
            for(int j=0; j<Req.size(); j++){
                relax(Req[j].w, Req[j].d);
            }
        }
        Req.clear();
        #pragma omp for
        for(int j=0; j<S.size(); j++){
            int v=S[j];
            for(int k=0; k<heavy[v].size(); k++){
                // int w = G[v][k].w;
                // req r;
                // r.w = w;
                // r.d = tent[v] + G[v][k].d;
                int w = heavy[v][k];
                req r;
                r.w = w;
                r.d = tent[v] + G[v][getidx(G[v], w)].d;
                #pragma omp critical
                {
                    Req.push_back(r);
                }
            }
        }
        #pragma omp for
        for(int j=0; j<Req.size(); j++){
            relax(Req[j].w, Req[j].d);
        }
        i++;
    }
}