#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 70

void matrix_multiply(int rank, int size, double A[SIZE][SIZE], double B[SIZE][SIZE], double *C_partial) {
    int start = rank * (SIZE / size);
    int end = (rank == size - 1) ? SIZE : (rank + 1) * (SIZE / size);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < SIZE; j++) {
            C_partial[(i - start) * SIZE + j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C_partial[(i - start) * SIZE + j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE * SIZE];
    double start_time, run_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > SIZE) {
        if (rank == 0) {
            fprintf(stderr, "Error: Too many MPI processes. Use up to %d processes.\n", SIZE);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        srand(0);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    MPI_Bcast(A, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);  
    start_time = omp_get_wtime(); 


    int local_rows = SIZE / size + (rank == size - 1 ? SIZE % size : 0);
    double *C_partial = (double *)malloc(local_rows * SIZE * sizeof(double));
    if (C_partial == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 

    matrix_multiply(rank, size, A, B, C_partial);

   
    MPI_Gather(C_partial, local_rows * SIZE, MPI_DOUBLE,
               C, local_rows * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    run_time = omp_get_wtime() - start_time; // Stop timing

    if (rank == 0) {
        printf("Matrix multiplication completed in %lf seconds.\n", run_time);
    }

    free(C_partial); 
    MPI_Finalize();
    return 0;
}