#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel function to compute the square root of each element
__global__ void squareRootKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(in[idx]);
    }
}

int main() {
    std::vector<int> arraySizes = {50000, 500000, 5000000, 50000000};

    for (int size : arraySizes) {
        // Host memory allocation
        std::vector<float> A(size);
        std::vector<float> C(size);

        // Initialize input vector A
        for (int i = 0; i < size; ++i) {
            A[i] = static_cast<float>(i); // Example initialization
        }

        // Device memory allocation
        float* dA;
        float* dC;
        cudaMalloc((void**)&dA, size * sizeof(float));
        cudaMalloc((void**)&dC, size * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(dA, A.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Configure thread block and grid dimensions
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        // Launch the CUDA kernel
        auto start = std::chrono::high_resolution_clock::now();
        squareRootKernel<<<numBlocks, blockSize>>>(dA, dC, size);
        cudaDeviceSynchronize(); // Wait for the kernel to finish
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Copy results from device to host
        cudaMemcpy(C.data(), dC, size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Time for array size " << size << ": " << duration.count() << " seconds" << std::endl;

        // Free device memory
        cudaFree(dA);
        cudaFree(dC);

        // You can optionally verify the results here
        // for (int i = 0; i < 10; ++i) {
        //     std::cout << "sqrt(" << A[i] << ") = " << C[i] << std::endl;
        // }
    }

    return 0;
}

Key Points:

Timing Output: The core of Part 3 is the std::cout line within the loop:

std::cout << "Time for array size " << size << ": " << duration.count() << " seconds" << std::endl;