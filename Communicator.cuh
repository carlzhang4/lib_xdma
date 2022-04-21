#ifndef communicator_cuh
#define communicator_cuh
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

void cu_all_reduce_float(float* start_addr, int size, unsigned int ** fpga_regs, unsigned int *params, cudaStream_t stream);
void cu_all_reduce_float_gloo(float* start_addr, int size, unsigned int ** fpga_regs, unsigned int *params);
#endif