#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 16)  

void daxpy_serial(double a, double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double a, double *X, double *Y, int rank, int size) {
    int local_n = N / size;  
    int start = rank * local_n;
    int end = start + local_n;

    for (int i = start; i < end; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *X, *Y;
    double a = 2.5;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    X = (double *)malloc(N * sizeof(double));
    Y = (double *)malloc(N * sizeof(double));

    
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            X[i] = rand() % 10;
            Y[i] = rand() % 10;
        }
    }

    
    
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start_time = MPI_Wtime();
    daxpy_parallel(a, X, Y, rank, size);
    end_time = MPI_Wtime();

    
    MPI_Gather(X + rank * (N / size), N / size, MPI_DOUBLE, X, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI DAXPY Time: %lf seconds\n", end_time - start_time);
    }

    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}