#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100  

int main(int argc, char *argv[]) {
    int rank, size;
    double local_sum = 0.0, global_sum = 0.0;
    double data[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = ARRAY_SIZE / size;  


    if (rank == 0) {
        srand(0);
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % 10; 
        }
    }


    double local_data[elements_per_proc];
    MPI_Scatter(data, elements_per_proc, MPI_DOUBLE,
                local_data, elements_per_proc, MPI_DOUBLE,
                0, MPI_COMM_WORLD);


    for (int i = 0; i < elements_per_proc; i++) {
        local_sum += local_data[i];
    }


    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Total sum of array elements: %lf\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}