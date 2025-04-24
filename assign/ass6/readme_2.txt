Part 2: CUDA Program to Compute Square Root

 File: sqrt_kernel.cu
 Description
CUDA kernel computes the square root of each element in a large array A and stores the result in array C.
Measures GPU computation time for various array sizes to show how performance scales with input size.
 How to Compile
nvcc -o sqrt_kernel sqrt_kernel.cu
▶ How to Run
bash
Copy
Edit
./sqrt_kernel
 Array Sizes Tested
50,000 elements

500,000 elements

5,000,000 elements

50,000,000 elements

The program prints execution time for each array size