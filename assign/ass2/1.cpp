
// Estimate the value of Pi using the Monte Carlo method and demonstrate basic MPI functions.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size, i;
    long long int num_points = 1000000;
    long long int local_num_points, local_count = 0, global_count = 0;
    double x, y;
    double pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_num_points = num_points / size;
    srand(time(NULL) + rank);

    for (i = 0; i < local_num_points; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if ((x * x + y * y) <= 1) {
            local_count++;
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = (4.0 * global_count) / num_points;
        printf("Estimated Pi value: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}
