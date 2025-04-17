#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024



__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];


__global__ void vectorAdd() {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];


    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }


    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

     
    cudaEventRecord(start, 0);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();

   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

     
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // in milliseconds

    
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

     
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_A[i], h_B[i], h_C[i]);
            return -1;
        }
    }

    printf("Vector addition successful.\n");
    printf("Kernel execution time: %f ms\n", elapsedTime);


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    int memClockRate = prop.memoryClockRate;
    int memBusWidth = prop.memoryBusWidth;


    double bandwidth = 2.0 * memClockRate * (memBusWidth / 8.0) / 1e6;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

   
    float totalBytes = N * (2 + 1) * sizeof(float); // 3 * N * 4 bytes
    double elapsedTimeSec = elapsedTime / 1000.0; // Convert ms to seconds
    double measuredBW = totalBytes / (elapsedTimeSec * 1e9); // GB/s
    printf("Measured Memory Bandwidth: %.2f GB/s\n", measuredBW);

     
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}