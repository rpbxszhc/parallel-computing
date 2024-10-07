#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define INF 1e9

typedef struct edge
{
    long w;
    int to;
    struct edge* next;
} edge;

typedef struct node
{
    edge* fe;
    int idx;
    bool tag;   // 确定node[i]是否为B[i]中原有的点
} node;

long max(long x, long y) {
    return (x > y) ? x : y;
}

int delta;
int bnum;
node* nodes;
long* d;
int V, E;
int v0;
int per;
int threads = 8;

void initial(int* V, int* E, int* v0) {
    FILE* f = fopen("case.in", "r");
    // fscanf(f, "%d %d %d", V, E, v0);
    scanf("%d %d %d", V, E, v0);
    per = (*V + threads - 1) / threads;
    nodes = (node*)malloc(sizeof(node) * *V);
    int* od = (int*)malloc(sizeof(int) * *V);
    d = (long*)malloc(sizeof(long) * *V);
    #pragma omp parallel for
    for(int i = 0; i < *V; i++) {
        nodes[i].fe = NULL;
        nodes[i].idx = -1;
        nodes[i].tag = false;
        od[i] = 0;
        d[i] = INF;
    }

    long max_w = 0;
    int max_od = 0;
    #pragma omp parallel
    {
        long local_max_w = 0;
        #pragma omp for nowait
        for(int i = 0; i < *E; i++) {
            int u, v;
            long w;
            // fscanf(f, "%d %d %ld", &u, &v, &w);
            scanf("%d %d %ld", &u, &v, &w);
            local_max_w = max(local_max_w, w);
            #pragma omp atomic
            od[u] += 1;
            
            edge* e = (edge*)malloc(sizeof(edge));
            e->w = w;
            e->to = v;
            #pragma omp critical
            {
                e->next = nodes[u].fe;
                nodes[u].fe = e;
            }

            //e = (edge*)malloc(sizeof(edge));
            //e->w = w;
            //e->to = u;
            //#pragma omp critical
            //{
            //    e->next = nodes[v].fe;
            //    nodes[v].fe = e;
            //}
        }
        #pragma omp critical
        {
            max_w = max(max_w, local_max_w);
        }
    }
    fclose(f);

    for(int i = 0; i < *V; i++)
        max_od = (int)max((long)max_od, (long)od[i]);

    delta = (max_w + max_od - 1) / max_od;
    bnum = (max_w + delta - 1) / delta;
    free(od);
    //printf("delta = %d,     bnum = %d,     max_w = %ld,     max_od = %d\n", delta, bnum, max_w, max_od);
}

void setTag(int min_idx) {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            if(nodes[i].idx == min_idx) nodes[i].tag = true;
        }
        #pragma omp barrier
    }
}

void unsetTag() {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            nodes[i].tag = false;
        }
        #pragma omp barrier
    }
}

void clearR() {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            if(nodes[i].idx == INF) nodes[i].idx = -1;
        }
        #pragma omp barrier
    }
}

void pushR() {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            if(nodes[i].tag) nodes[i].idx = INF;    // 记R为标号为INF的桶
        }
        #pragma omp barrier
    }
}

int findMinIdx() {
    int min_idx = INF;
    #pragma omp parallel for
    for(int i = 0; i < V; i++) {
        int cur_idx = nodes[i].idx;
        if(cur_idx != -1)  min_idx = (min_idx < cur_idx) ? min_idx : cur_idx;
    }
    #pragma omp barrier
    return min_idx;
}

void relaxLight() {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            if(!nodes[i].tag)   continue;
            edge* e = nodes[i].fe;
            while(e) {
                if(e->w <= delta) {
                    int cur_d = e->w + d[i];
                    #pragma omp critical
                    {
                        if(cur_d < d[e->to]) {
                            nodes[e->to].idx = cur_d / delta;
                            d[e->to] = cur_d;
                        }
                    }
                }
                
                e = e->next;
            }
        }
        #pragma omp barrier
    }
}

void relaxHeavy() {
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        for(int i = per * t; i < per * (t+1) && i < V; i++) {
            if(nodes[i].idx != INF) continue;
            edge* e = nodes[i].fe;
            while(e) {
                if(e->w > delta) {
                    int cur_d = e->w + d[i];
                    #pragma omp critical
                    {
                        if(cur_d < d[e->to]) {
                            nodes[e->to].idx = cur_d / delta;
                            d[e->to] = cur_d;
                        }
                    }
                }
                
                e = e->next;
            }
        }
        #pragma omp barrier
    }
}

bool allEmpty() {
    int flag = 1;
    #pragma omp parallel for
    for(int i = 0; i < V; i++)
        if(nodes[i].idx != -1 && nodes[i].idx != INF)  flag = false;
    #pragma omp barrier
    return flag;
}

bool isEmpty(int cur_idx) {
    bool flag = true;
    #pragma omp parallel for
    for(int i = 0; i < V; i++)
        if(nodes[i].idx == cur_idx)  flag = false;
    #pragma omp barrier
    return flag;
}

int main() {
    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();
    initial(&V, &E, &v0);
    double t2 = omp_get_wtime();
    d[v0] = 0;
    nodes[v0].idx = 0;
    int i = 0, j;

    while(!allEmpty()) {
        clearR();
        int cur_idx = findMinIdx();
        if(cur_idx == INF)  break;
        while(1) {
            setTag(cur_idx);
            pushR();
            relaxLight();
            unsetTag();
            if(isEmpty(cur_idx))    break;
        }
        relaxHeavy();
    }
    double t1 = omp_get_wtime();
    // printf("%lf\n", t1-t0);
    // printf("%lf\n", t2-t0);

    // FILE* f = fopen("case1.out", "w");
    // for(int i = 0; i < V; i++) {
    //     if(d[i] == INF) fprintf(f, "INF ");
    //     else    fprintf(f, "%ld ", d[i]);
    // }

    for(int i = 0; i < V; i++) {
       if(d[i] == INF) printf("INF ");
       else    printf("%ld ", d[i]);
    }

    free(d);
    for(int i = 0; i < V; i++) {
        edge* t = nodes[i].fe;
        while(t) {
            edge* n = t->next;
            free(t);
            t = n;
        }
    }
    free(nodes);
}