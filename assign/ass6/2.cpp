// sqrt_kernel.cu
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <chrono>

__global__ void computeSqrt(float* A, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = sqrtf(A[idx]);
    }
}

void runSqrtKernel(int N) {
    float *h_A, *h_C;
    float *d_A, *d_C;

    size_t size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) h_A[i] = i + 1;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();
    computeSqrt<<<blocks, threads>>>(d_A, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Square Root Time for N = " << N << ": " << elapsed.count() << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_A); free(h_C);
    cudaFree(d_A); cudaFree(d_C);
}

int main() {
    runSqrtKernel(50000);
    runSqrtKernel(500000);
    runSqrtKernel(5000000);
    runSqrtKernel(50000000);
    return 0;
}