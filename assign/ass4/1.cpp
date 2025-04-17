#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void compute_sums(int *iterative_sum, int *formula_sum) {
    int tid = threadIdx.x;

    if (tid == 0) {
        
        int sum = 0;
        for (int i = 1; i <= N; ++i) {
            sum += i;
        }
        *iterative_sum = sum;
    } else if (tid == 1) {
        
        *formula_sum = N * (N + 1) / 2;
    }
}

int main() {
    int h_iterative_sum = 0;
    int h_formula_sum = 0;
    int *d_iterative_sum, *d_formula_sum;

  
    cudaMalloc((void**)&d_iterative_sum, sizeof(int));
    cudaMalloc((void**)&d_formula_sum, sizeof(int));


    compute_sums<<<1, 2>>>(d_iterative_sum, d_formula_sum);

   
    cudaMemcpy(&h_iterative_sum, d_iterative_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_formula_sum, d_formula_sum, sizeof(int), cudaMemcpyDeviceToHost);

    
    printf("Sum using iterative approach: %d\n", h_iterative_sum);
    printf("Sum using formula approach: %d\n", h_formula_sum);

    
    cudaFree(d_iterative_sum);
    cudaFree(d_formula_sum);

    return 0;
}