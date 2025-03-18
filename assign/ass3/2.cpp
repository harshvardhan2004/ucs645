#include <mpi.h>
#include <stdio.h>

#define NUM_STEPS 100000

int main(int argc, char *argv[]) {
    int rank, size, i;
    double step, local_sum = 0.0, pi = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    step = 1.0 / (double)NUM_STEPS;
    MPI_Bcast(&step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = rank; i < NUM_STEPS; i += size) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = step * pi;
        printf("Approximated π: %lf\n", pi);
    }

    MPI_Finalize();
    return 0;
}