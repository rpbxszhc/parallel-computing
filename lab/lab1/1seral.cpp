#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <omp.h>

using namespace std;

void luDecomposition(vector<vector<double>>& A, vector<vector<int>>& L, vector<vector<int>>& U) {
    int n = A.size();

    // Initialize L and U matrices
    L = vector<vector<int>>(n, vector<int>(n, 0));
    U = vector<vector<int>>(n, vector<int>(n, 0));
    int num_threads = 8;
    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        for(int k = 0; k < n; k++){
            if(k % num_threads == id) {
                for(int i = k + 1; i < n; i++) {
                    A[i][k] /= A[k][k]; 
                }
            }
            #pragma omp barrier
            #pragma omp for schedule(dynamic,1)
            for(int i = k + 1; i < n; i++){
                for(int j = k + 1; j < n; j++){
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
        #pragma omp for schedule(dynamic,1)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(i > j){
                    L[i][j] = A[i][j];
                }else{
                    U[i][j] = A[i][j];
                }
            }
            L[i][i] = 1;
        }
    }
}

int main() {
    
    int n;
    cin>>n;
    vector<vector<double>> A(n, vector<double>(n));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            cin>>A[i][j];
        }
    }
    vector<vector<int>> L, U;

    luDecomposition(A, L, U);
    
    for (const auto& row : L) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    for (const auto& row : U) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
