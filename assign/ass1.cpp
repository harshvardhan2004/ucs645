#include <mpi.h>
#include <iostream>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For seeding random number generator

#define MIN_POSITION 0   // Minimum position in the domain
#define MAX_POSITION 100 // Maximum position in the domain
#define NUM_WALKERS 5    // Number of walkers per process
#define MAX_STEPS 20     // Maximum number of steps each walker takes
#define TAG 0            // MPI message tag

using namespace std;

// Structure to store walker data
struct Walker {
    int position;
    int steps_left;
};

int main(int argc, char** argv) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Seed random generator differently for each process
    srand(time(0) + rank);

    // Define the range each process handles
    int local_min = rank * (MAX_POSITION / size);
    int local_max = (rank + 1) * (MAX_POSITION / size) - 1;

    // Initialize walkers for each process
    Walker walkers[NUM_WALKERS];
    for (int i = 0; i < NUM_WALKERS; i++) {
        walkers[i].position = local_min; // Start from the beginning of local range
        walkers[i].steps_left = (rand() % MAX_STEPS) + 1; // Random step count
    }

    // Process Walkers
    for (int i = 0; i < NUM_WALKERS; i++) {
        while (walkers[i].steps_left > 0) {
            walkers[i].position++;

            // Check if the walker exceeds the local range
            if (walkers[i].position > local_max) {
                if (rank < size - 1) {
                    // Send walker to the next process
                    MPI_Send(&walkers[i], 2, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);
                    break; // Stop processing this walker
                } else {
                    // Wrap-around to MIN_POSITION if at the last process
                    walkers[i].position = MIN_POSITION;
                }
            }

            walkers[i].steps_left--;
        }
    }

    // Receiving Walkers
    MPI_Status status;
    Walker incoming_walker;
    while (MPI_Recv(&incoming_walker, 2, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status) == MPI_SUCCESS) {
        // Continue processing the received walker
        while (incoming_walker.steps_left > 0) {
            incoming_walker.position++;

            if (incoming_walker.position > local_max) {
                if (rank < size - 1) {
                    // Send to the next process
                    MPI_Send(&incoming_walker, 2, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);
                    break;
                } else {
                    // Wrap-around at last process
                    incoming_walker.position = MIN_POSITION;
                }
            }

            incoming_walker.steps_left--;
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}