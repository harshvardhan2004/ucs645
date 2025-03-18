#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 8  // Number of elements

int main(int argc, char *argv[]) {
    int rank, size;
    int local_value, prefix_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    local_value = rank + 1; 

    
    
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("Process %d: Local Value = %d, Prefix Sum = %d\n", rank, local_value, prefix_sum);

    MPI_Finalize();
    return 0;
}