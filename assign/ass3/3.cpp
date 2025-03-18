#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NUM 100  

int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size, num, flag;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {  // Master
        for (num = 2; num <= MAX_NUM; num++) {
            int worker;
            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&num, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
        }
        
        // Send termination signal (-1)
        for (int i = 1; i < size; i++) {
            MPI_Recv(&num, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        printf("Prime finding complete.\n");

    } else {  // Workers
        while (1) {
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num == -1) break;

            flag = is_prime(num) ? num : -num;
            MPI_Send(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}