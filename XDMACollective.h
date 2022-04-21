#ifndef XDMA_COLLECTIVE_H
#define XDMA_COLLECTIVE_H

#include <iostream>
#include <iomanip>
#include <vector>

#include <gdrapi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "XDMA.h"
#include "XDMAController.h"

using namespace std;

typedef struct{
	size_t start_page;
	size_t end_page;//is same with start_page when there is only one page
	FpgaPageTable tlb;
}TLBTable;


#define ASSERT(x)                                                       \
    do                                                                  \
        {                                                               \
            if (!(x))                                                   \
                {                                                       \
                    fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
                    exit(EXIT_FAILURE);                                 \
                }                                                       \
        } while (0)

#define ASSERTDRV(stmt)				\
    do                                          \
        {                                       \
            CUresult result = (stmt);           \
            if (result != CUDA_SUCCESS) {       \
                const char *_err_name;          \
                cuGetErrorName(result, &_err_name); \
                fprintf(stderr, "CUDA error: %s\n", _err_name);   \
            }                                   \
            ASSERT(CUDA_SUCCESS == result);     \
        } while (0)

#define ASSERT_EQ(P, V) ASSERT((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT(!((P) == (V)))
#define BREAK_IF_NEQ(P, V) if((P) != (V)) break


#define ONCE_RUN(code) {                                    \
    static int _done=0;                                     \
    if (!_done) {                                           \
            code                                            \
			_done=1;										\
    }                                                       \
}

#define ErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
void xdma_all_reduce_gloo(void* send_buf, size_t number, int dtype_size,int rank, int num_machine);
void xdma_all_reduce_test(void* send_buf, void* recv_buf, size_t number,int dtype, cudaStream_t stream,int rank, int num_machine);
int xdma_all_reduce(size_t addr_start,size_t number,int size);
int gdr_init();
unsigned int* map_reg_4(int reg,fpga::XDMAController* controller);
#endif