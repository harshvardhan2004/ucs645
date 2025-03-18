#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 20        
#define MAX_ITER 500 
#define EPSILON 0.01 


void initialize_grid(double grid[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = 0.0; 
        }
    }
    

    for (int i = 0; i < N; i++)
    {
        grid[0][i] = 100.0;    
        grid[N - 1][i] = 100.0; 
    }
}


void compute_heat_distribution(double local_grid[][N], double new_local_grid[][N], int start, int end)
{
    for (int i = start; i < end; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            new_local_grid[i][j] = 0.25 * (local_grid[i - 1][j] + local_grid[i + 1][j] +
                                           local_grid[i][j - 1] + local_grid[i][j + 1]);
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    double grid[N][N], new_grid[N][N];
    int local_start, local_end;
    double max_diff, global_diff;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size; 


    local_start = rank * rows_per_proc;
    local_end = (rank + 1) * rows_per_proc;

    if (rank == 0)
    {
        initialize_grid(grid); 


    MPI_Bcast(grid, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int iter = 0;
    do
    {
        max_diff = 0.0;


        if (rank > 0)
        { 
            MPI_Sendrecv(&grid[local_start][0], N, MPI_DOUBLE, rank - 1, 0,
                         &grid[local_start - 1][0], N, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1)
        { 
            MPI_Sendrecv(&grid[local_end - 1][0], N, MPI_DOUBLE, rank + 1, 0,
                         &grid[local_end][0], N, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        
        compute_heat_distribution(grid, new_grid, local_start, local_end);

       
        for (int i = local_start; i < local_end; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                double diff = fabs(new_grid[i][j] - grid[i][j]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                }
                grid[i][j] = new_grid[i][j]; 
            }
        }


        MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        iter++;
    } while (global_diff > EPSILON && iter < MAX_ITER);

    
    double final_grid[N][N];

    MPI_Gather(&grid[local_start][0], rows_per_proc * N, MPI_DOUBLE,
               &final_grid[0][0], rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    if (rank == 0)
    {
        printf("Final Heat Distribution:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%6.2f ", final_grid[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}