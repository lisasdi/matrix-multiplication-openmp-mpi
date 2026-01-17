#!/bin/bash

# Script de compilation et exécution - Matrix-Matrix Multiplication
# Adapté pour WSL avec g++, OpenMP et OpenMPI

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Matrix-Matrix Multiplication Project ===${NC}"
echo "Sequential, OpenMP, MPI, and Hybrid implementations"

# Créer un fichier CSV pour les résultats
echo "version,time_ms,gflops,throughput_gb_s,num_threads,num_ranks" > metrics.csv

# 1. Compilation séquentielle
echo -e "\n${YELLOW}[1] Compiling Sequential version...${NC}"
g++ -O3 -march=native -std=c++17 01_sequential.cpp -o seq
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sequential compiled${NC}"
else
    echo -e "${RED}✗ Sequential compilation failed${NC}"
    exit 1
fi

# 2. Compilation OpenMP
echo -e "\n${YELLOW}[2] Compiling OpenMP version...${NC}"
g++ -O3 -march=native -std=c++17 -fopenmp 02_openmp.cpp -o omp
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ OpenMP compiled${NC}"
else
    echo -e "${RED}✗ OpenMP compilation failed${NC}"
    exit 1
fi

# 3. Compilation MPI (utiliser mpicc/mpicxx si disponible)
echo -e "\n${YELLOW}[3] Compiling MPI version...${NC}"
if command -v mpicxx &> /dev/null; then
    mpicxx -O3 -march=native -std=c++17 03_mpi.cpp -o mpi
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MPI compiled${NC}"
    else
        echo -e "${RED}✗ MPI compilation failed${NC}"
        exit 1
    fi
elif command -v g++ &> /dev/null; then
    echo -e "${YELLOW}  mpicxx not found, trying g++ with MPI headers...${NC}"
    g++ -O3 -march=native -std=c++17 -I/usr/include/mpi 03_mpi.cpp -o mpi -lmpi -lmpi_cxx
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MPI compiled with g++${NC}"
    else
        echo -e "${RED}✗ MPI compilation failed${NC}"
        echo -e "${YELLOW}  Skipping MPI execution${NC}"
        MPI_FAILED=1
    fi
fi

# 4. Compilation Hybride
echo -e "\n${YELLOW}[4] Compiling Hybrid version...${NC}"
if command -v mpicxx &> /dev/null; then
    mpicxx -O3 -march=native -std=c++17 -fopenmp 04_hybrid.cpp -o hybrid
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Hybrid compiled${NC}"
    else
        echo -e "${RED}✗ Hybrid compilation failed${NC}"
        exit 1
    fi
elif command -v g++ &> /dev/null; then
    echo -e "${YELLOW}  mpicxx not found, trying g++ with MPI headers...${NC}"
    g++ -O3 -march=native -std=c++17 -fopenmp -I/usr/include/mpi 04_hybrid.cpp -o hybrid -lmpi -lmpi_cxx
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Hybrid compiled with g++${NC}"
    else
        echo -e "${RED}✗ Hybrid compilation failed${NC}"
        echo -e "${YELLOW}  Skipping Hybrid execution${NC}"
        HYBRID_FAILED=1
    fi
fi

# Exécution des programmes
echo -e "\n${YELLOW}=== EXECUTION ===${NC}"

# 1. Exécution séquentielle
echo -e "\n${YELLOW}[1] Running Sequential...${NC}"
./seq
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sequential completed${NC}"
else
    echo -e "${RED}✗ Sequential failed${NC}"
fi

# 2. Exécution OpenMP
echo -e "\n${YELLOW}[2] Running OpenMP...${NC}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./omp
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ OpenMP completed${NC}"
else
    echo -e "${RED}✗ OpenMP failed${NC}"
fi

# 3. Exécution MPI
if [ $MPI_FAILED -ne 1 ]; then
    echo -e "\n${YELLOW}[3] Running MPI with 2 ranks...${NC}"
    if command -v mpirun &> /dev/null; then
        mpirun -np 2 ./mpi
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ MPI completed${NC}"
        else
            echo -e "${RED}✗ MPI failed${NC}"
        fi
    else
        echo -e "${YELLOW}  mpirun not found, skipping MPI execution${NC}"
    fi
fi

# 4. Exécution Hybride
if [ $HYBRID_FAILED -ne 1 ]; then
    echo -e "\n${YELLOW}[4] Running Hybrid with 2 ranks...${NC}"
    if command -v mpirun &> /dev/null; then
        mpirun -np 2 ./hybrid
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Hybrid completed${NC}"
        else
            echo -e "${RED}✗ Hybrid failed${NC}"
        fi
    else
        echo -e "${YELLOW}  mpirun not found, skipping Hybrid execution${NC}"
    fi
fi

echo -e "\n${YELLOW}=== RESULTS ===${NC}"
if [ -f metrics.csv ]; then
    echo "Metrics saved to metrics.csv"
    echo -e "\n${GREEN}CSV Content:${NC}"
    cat metrics.csv
else
    echo -e "${RED}No metrics file found${NC}"
fi

echo -e "\n${GREEN}✓ All tests completed!${NC}"
echo "Run 'python3 plot_results.py' to generate graphs"