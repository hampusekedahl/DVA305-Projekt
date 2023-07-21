# Matrix Multiplication Benchmark

This repository contains C code for benchmarking matrix multiplication using three different implementations: standard version, SSE version, and AVX version. The benchmark measures the performance of each implementation and compares the results to ensure correctness.

## Matrix Multiplication Implementations

1. version1: Standard version of matrix multiplication using three nested loops for element-wise multiplication and summation.
   
3. SSE: SSE version of matrix multiplication using SIMD (Single Instruction, Multiple Data) intrinsics to accelerate the computation for 4 elements at a time.
   
5. AVX: AVX version of matrix multiplication using AVX SIMD intrinsics to accelerate the computation for 8 elements at a time.

## Results

| Matrix Size | Execution Time in seconds - SSE | Execution Time in seconds - AVX |
|:-----------:|:-------------------------------:|:-------------------------------:|
|   480x480   |              0.036              |              0.037              |
|   960x960   |              0.353              |              0.314              |
|  1440x1440  |              1.759              |              0.998              |
|  1920x1920  |              5.517              |              2.651              |
|  2400x2400  |              11.094             |              5.125              |

## Note

Note: If using Visual Studio, the console window will not close immediately after running the program. This allows you to see the output.
