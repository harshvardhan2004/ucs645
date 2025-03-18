#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 100  

int main(int argc, char *argv[]) {
    int rank, size;
    double local_dot = 0.0, global_dot = 0.0;
    double A[VECTOR_SIZE], B[VECTOR_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = VECTOR_SIZE / size;  

   
    if (rank == 0) {
        srand(0);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    }


    double local_A[elements_per_proc], local_B[elements_per_proc];
    MPI_Scatter(A, elements_per_proc, MPI_DOUBLE, local_A, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, elements_per_proc, MPI_DOUBLE, local_B, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    for (int i = 0; i < elements_per_proc; i++) {
        local_dot += local_A[i] * local_B[i];
    }


    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Dot Product: %lf\n", global_dot);
    }

    MPI_Finalize();
    return 0;
}