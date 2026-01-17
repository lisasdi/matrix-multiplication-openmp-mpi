#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>
#include <omp.h>

using namespace std;

const int M = 2000;
const int K = 2000;
const int N = 2000;

void initialize_matrices(double* A, double* B, double* C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = (double)rand() / RAND_MAX;
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (double)rand() / RAND_MAX;
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
        }
    }
}

void matrix_multiply_openmp(double* A, double* B, double* C, int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Deux boucles parallèles imbriquées (i, j) avec k séquentiel
    #pragma omp parallel for collapse(2) schedule(dynamic, 32)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void save_metrics(const char* version, double time_ms, int num_threads, int num_ranks) {
    ofstream file("metrics.csv", ios::app);
    double flops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;
    double throughput = (M * K + K * N + M * N) * sizeof(double) / (time_ms / 1000.0) / 1e9;
    file << version << "," << time_ms << "," << flops << "," << throughput 
         << "," << num_threads << "," << num_ranks << "\n";
    file.close();
}

int main() {
    int max_threads = omp_get_max_threads();
    
    cout << "=== Matrix-Matrix Multiplication (OpenMP) ===" << endl;
    cout << "Matrice A: " << M << " x " << K << endl;
    cout << "Matrice B: " << K << " x " << N << endl;
    cout << "Matrice C: " << M << " x " << N << endl;
    cout << "Max threads available: " << max_threads << endl;
    cout << "Total memory: ~" << (M * K + K * N + M * N) * sizeof(double) / 1e6 << " MB" << endl;

    double* A = new double[M * K];
    double* B = new double[K * N];
    double* C = new double[M * N];

    cout << "\nInitializing matrices..." << endl;
    initialize_matrices(A, B, C);

    // Test avec différents nombres de threads
    cout << "\n=== Testing different thread counts ===" << endl;
    for (int num_threads = 1; num_threads <= max_threads; num_threads++) {
        cout << "\nThread count: " << num_threads << endl;
        
        // Réinitialiser C
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = 0.0;
            }
        }
        
        auto start = chrono::high_resolution_clock::now();
        matrix_multiply_openmp(A, B, C, num_threads);
        auto end = chrono::high_resolution_clock::now();

        double time_ms = chrono::duration<double, milli>(end - start).count();
        double flops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;

        cout << "  Time: " << time_ms << " ms" << endl;
        cout << "  GFLOPS: " << flops << " GFLOPS" << endl;
        cout << "  Result: C[0][0] = " << C[0] << endl;

        string version = "OpenMP_" + to_string(num_threads) + "T";
        save_metrics(version.c_str(), time_ms, num_threads, 1);
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}