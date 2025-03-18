#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 6  

void print_matrix(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5.1f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[SIZE][SIZE], B[SIZE][SIZE];
    double local_A[SIZE][SIZE], local_B[SIZE][SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = SIZE / size; 
    

    if (rank == 0) {
        printf("Original Matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = i * SIZE + j + 1;  
            }
        }
        print_matrix(A);
    }

    
    
    MPI_Scatter(A, rows_per_proc * SIZE, MPI_DOUBLE, local_A, rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < SIZE; j++) {
            local_B[j][i + rank * rows_per_proc] = local_A[i][j]; 
        }
    }

    
    MPI_Gather(local_B, rows_per_proc * SIZE, MPI_DOUBLE, B, rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   
    if (rank == 0) {
        printf("Transposed Matrix:\n");
        print_matrix(B);
    }

    MPI_Finalize();
    return 0;
}