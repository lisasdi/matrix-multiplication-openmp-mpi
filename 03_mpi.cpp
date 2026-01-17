#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>
#include <mpi.h>

using namespace std;

const int M = 2000;
const int K = 2000;
const int N = 2000;

void initialize_matrices_global(double* A, double* B, double* C) {
    // Initialiser uniquement sur le rank 0
    for (int i = 0; i < M * K; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (double)rand() / RAND_MAX;
    }
    memset(C, 0, M * N * sizeof(double));
}

void matrix_multiply_mpi(double* A_local, double* B, double* C_local,
                         int local_rows, int rank, int num_ranks) {
    // Chaque rank multiplie ses lignes locales par B complète
    // C_local[i][j] = sum(A_local[i][k] * B[k][j]) for k = 0 to K-1
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A_local[i * K + k] * B[k * N + j];
            }
            C_local[i * N + j] = sum;
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

int main(int argc, char** argv) {
    int rank, num_ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Vérifier que M est divisible par num_ranks
    if (M % num_ranks != 0) {
        if (rank == 0) {
            cerr << "Error: M (" << M << ") must be divisible by num_ranks (" 
                 << num_ranks << ")" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int local_rows = M / num_ranks;

    if (rank == 0) {
        cout << "=== Matrix-Matrix Multiplication (MPI) ===" << endl;
        cout << "Matrice A: " << M << " x " << K << endl;
        cout << "Matrice B: " << K << " x " << N << endl;
        cout << "Matrice C: " << M << " x " << N << endl;
        cout << "MPI Ranks: " << num_ranks << endl;
        cout << "Local rows per rank: " << local_rows << endl;
        cout << "Total memory: ~" << (M * K + K * N + M * N) * sizeof(double) / 1e6 << " MB" << endl;
    }

    // Allouer les données locales
    double* A_local = new double[local_rows * K];
    double* B = new double[K * N];
    double* C_local = new double[local_rows * N];

    // Sur rank 0: créer les matrices complètes
    double* A_global = nullptr;
    double* C_global = nullptr;
    if (rank == 0) {
        A_global = new double[M * K];
        C_global = new double[M * N];
        cout << "\nInitializing matrices..." << endl;
        initialize_matrices_global(A_global, B, C_global);
    }

    // Initialiser B sur tous les ranks
    if (rank == 0) {
        for (int i = 0; i < K * N; i++) {
            B[i] = (rank == 0) ? B[i] : 0.0;
        }
    }
    
    // Broadcast B à tous les ranks
    MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A par lignes
    MPI_Scatter(A_global, local_rows * K, MPI_DOUBLE,
                A_local, local_rows * K, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Barrier pour synchroniser avant le calcul
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Computing matrix multiplication (MPI)..." << endl;
    }

    auto start = chrono::high_resolution_clock::now();
    
    matrix_multiply_mpi(A_local, B, C_local, local_rows, rank, num_ranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();

    // Gather les résultats
    MPI_Gather(C_local, local_rows * N, MPI_DOUBLE,
               C_global, local_rows * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double time_ms = chrono::duration<double, milli>(end - start).count();
        double flops = 2.0 * M * N * K / (time_ms / 1000.0) / 1e9;

        cout << "\n=== RESULTS ===" << endl;
        cout << "Time: " << time_ms << " ms" << endl;
        cout << "GFLOPS: " << flops << " GFLOPS" << endl;
        cout << "Verification: C[0][0] = " << C_global[0] << endl;

        string version = "MPI_" + to_string(num_ranks) + "R";
        save_metrics(version.c_str(), time_ms, 1, num_ranks);

        delete[] A_global;
        delete[] C_global;
    }

    delete[] A_local;
    delete[] B;
    delete[] C_local;

    MPI_Finalize();
    return 0;
}