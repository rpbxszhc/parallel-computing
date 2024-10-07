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
    // omp_set_num_threads(8);
    // Initialize L and U matrices
    L = vector<vector<int>>(n, vector<int>(n, 0));
    U = vector<vector<int>>(n, vector<int>(n, 0));
    #pragma omp parallel num_threads(8)
    {
        for(int k = 0; k < n; k++){
            #pragma omp for
            for(int i = k + 1; i < n; i++) {
                A[i][k] /= A[k][k]; 
            }
            #pragma omp for collapse(2)
            for(int i = k + 1; i < n; i++){
                for(int j = k + 1; j < n; j++){
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
        #pragma omp for collapse(2)
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(i > j){
                    L[i][j] = A[i][j];
                }else{
                    U[i][j] = A[i][j];
                }
            }
        }
        #pragma omp for
        for(int i = 0; i < n; i++){
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
