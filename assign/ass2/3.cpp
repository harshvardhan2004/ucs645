#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 16  


void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

//Bubble Sort
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Odd-Even Sort using MPI
void parallel_odd_even_sort(int *local_data, int local_n, int rank, int size) {
    int phase;
    int partner;
    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) { 

            if (rank % 2 == 0) {
                partner = rank + 1;
            } else {
                partner = rank - 1;
            }
        } else {
            
            if (rank % 2 == 0) {
                partner = rank - 1;
            } else {
                partner = rank + 1;
            }
        }

        if (partner >= 0 && partner < size) {
            int recv_data[local_n];
            MPI_Sendrecv(local_data, local_n, MPI_INT, partner, 0,
                         recv_data, local_n, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Merge
            int merged[2 * local_n];
            for (int i = 0; i < local_n; i++) merged[i] = local_data[i];
            for (int i = 0; i < local_n; i++) merged[local_n + i] = recv_data[i];

            bubble_sort(merged, 2 * local_n);


            if (rank < partner) {
                for (int i = 0; i < local_n; i++) local_data[i] = merged[i];
            } else {
                for (int i = 0; i < local_n; i++) local_data[i] = merged[local_n + i];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *global_data = NULL;
    int local_n;
    int *local_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = SIZE / size;  


    if (rank == 0) {
        global_data = (int *)malloc(SIZE * sizeof(int));
        srand(0);
        for (int i = 0; i < SIZE; i++) {
            global_data[i] = rand() % 100;
        }
        printf("Unsorted Array: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", global_data[i]);
        }
        printf("\n");
    }


    local_data = (int *)malloc(local_n * sizeof(int));


    MPI_Scatter(global_data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);


    bubble_sort(local_data, local_n);


    parallel_odd_even_sort(local_data, local_n, rank, size);


    MPI_Gather(local_data, local_n, MPI_INT, global_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Sorted Array: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", global_data[i]);
        }
        printf("\n");
        free(global_data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}