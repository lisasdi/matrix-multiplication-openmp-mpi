#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>

using namespace std;

// Dimensions des matrices
const int M = 2000;  // Nombre de lignes de A et C
const int K = 2000;  // Nombre de colonnes de A, lignes de B
const int N = 2000;  // Nombre de colonnes de B et C

// Matrice A: M × K
// Matrice B: K × N
// Matrice C: M × N

void initialize_matrices(double* A, double* B, double* C) {
    // Initialiser A et B avec des valeurs aléatoires
    for (int i = 0; i < M * K; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (double)rand() / RAND_MAX;
    }
    // C initialisé à 0
    memset(C, 0, M * N * sizeof(double));
}

void matrix_multiply_sequential(double* A, double* B, double* C) {
    // C = A × B
    // C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to K-1
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
    double flops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;  // GFLOPS
    double throughput = (M * K + K * N + M * N) * sizeof(double) / (time_ms / 1000.0) / 1e9;  // GB/s
    file << version << "," << time_ms << "," << flops << "," << throughput 
         << "," << num_threads << "," << num_ranks << "\n";
    file.close();
}

int main() {
    cout << "=== Matrix-Matrix Multiplication (Sequential) ===" << endl;
    cout << "Matrice A: " << M << " x " << K << endl;
    cout << "Matrice B: " << K << " x " << N << endl;
    cout << "Matrice C: " << M << " x " << N << endl;
    cout << "Total memory: ~" << (M * K + K * N + M * N) * sizeof(double) / 1e6 << " MB" << endl;

    // Allouer les matrices
    double* A = new double[M * K];
    double* B = new double[K * N];
    double* C = new double[M * N];

    cout << "\nInitializing matrices..." << endl;
    initialize_matrices(A, B, C);

    cout << "Computing matrix multiplication (sequential)..." << endl;
    auto start = chrono::high_resolution_clock::now();
    matrix_multiply_sequential(A, B, C);
    auto end = chrono::high_resolution_clock::now();

    double time_ms = chrono::duration<double, milli>(end - start).count();
    double flops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;  // GFLOPS

    cout << "\n=== RESULTS ===" << endl;
    cout << "Time: " << time_ms << " ms" << endl;
    cout << "GFLOPS: " << flops << " GFLOPS" << endl;
    cout << "Verification: C[0][0] = " << C[0] << endl;

    // Sauvegarder les métriques
    save_metrics("Sequential", time_ms, 1, 1);

    // Nettoyer
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}