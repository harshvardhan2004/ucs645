This project implements and compares two versions of the Merge Sort algorithm for sorting an array of size n = 1000:

Parallel Merge Sort with Pipelining: Utilizes CPU-based parallelism with pipelining techniques to enhance performance.

Parallel Merge Sort using CUDA: Leverages GPU acceleration to perform merge sort operations in parallel using NVIDIA's CUDA framework.

The project also includes a performance comparison between the two methods to evaluate efficiency and speedup achieved through GPU acceleration.

Prerequisites Hardware: NVIDIA GPU with CUDA support.

Software:

CUDA Toolkit installed.

C++ compiler compatible with CUDA (e.g., nvcc).

For pipelined version: Support for multithreading (e.g., OpenMP or C++11 threads).

Compilation CUDA Version bash Copy Edit nvcc -O3 -o cuda_merge_sort cuda_merge_sort.cu Pipelined CPU Version bash Copy Edit g++ -O3 -fopenmp -o pipelined_merge_sort pipelined_merge_sort.cpp Note: Ensure that your compiler supports OpenMP if you're using it for pipelining.

Execution CUDA Version bash Copy Edit ./cuda_merge_sort Pipelined CPU Version bash Copy Edit ./pipelined_merge_sort Implementation Details

Parallel Merge Sort with Pipelining Approach: Implements merge sort on the CPU using parallel threads. Pipelining is achieved by overlapping the sorting of subarrays with the merging process.
Techniques:

Divides the array into subarrays.

Sorts subarrays in parallel.

Merges sorted subarrays in a pipelined fashion to utilize CPU cores efficiently.

Parallel Merge Sort using CUDA Approach: Implements merge sort on the GPU using CUDA to exploit massive parallelism.
Techniques:

Transfers data to GPU memory.

Launches CUDA kernels to perform sorting and merging operations in parallel.

Transfers the sorted array back to host memory.

Performance Comparison After executing both versions, the program outputs the execution time for each method. This allows for a direct comparison of performance between CPU-based pipelined merge sort and GPU-accelerated merge sort.

Sample Output:

sql Copy Edit Pipelined Merge Sort Execution Time: 45.23 ms CUDA Merge Sort Execution Time: 12.67 ms Speedup Achieved with CUDA: 3.57x
