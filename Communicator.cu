#include "Communicator.cuh"

using namespace std;

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

__device__ volatile size_t count = 0;

__device__ int cu_sleep(double seconds){
	size_t s = clock64();
	while( clock64()-s<size_t(3000000000.0*seconds)){//1s = 3000000000
		// if(seconds==0.01){
		// 	printf("here is the %ld %ld\n",s,clock64());
		// }
	};
	return int(s);
}

__device__ void write_bypass(volatile unsigned int *dev_addr,unsigned int *data){
	int index = threadIdx.x;
	__syncthreads();
	if(index<16){
		dev_addr[index] = data[index];	
	}
	__syncthreads();
}

__global__ void kernel_all_reduce_float(	volatile float* start_addr, 
											int size,
											volatile unsigned int* reg_bypass,
											volatile unsigned int* reg_response,

											unsigned int total_length,
											unsigned int each_client_length,
											unsigned int addr_start_64B_low,
											unsigned int addr_start_64B_high,
											unsigned int head_length,
											unsigned int tail_length,
											unsigned int dtype,
											unsigned int *reg_rx_overflow
											){

	int index = threadIdx.x;
	__shared__ unsigned int data[16];
	unsigned int last_response = 0;

	size_t addr = (size_t)start_addr;
	addr = addr - 0x40;
	unsigned int* aligned_start = (unsigned int*) addr;

	data[0]	= addr_start_64B_low;
	data[1]	= addr_start_64B_high;
	data[2]	= each_client_length;
	data[3]	= total_length;
	data[4]	= tail_length;
	data[5]	= head_length;
	data[6] = dtype;
	if(index==0){
		last_response = *reg_response;
		// printf("dtype:%d\n",dtype);
		// printf("last_response:%d\n",last_response);
		// for(int i=0;i<128;i++){
		// 	if(i%16==0){
		// 		printf("\n");
		// 	}
		// 	printf("%x ",aligned_start[i]);
		// }
		// printf("\n");
		// cu_sleep(1);
	}
	__syncthreads();
	write_bypass(reg_bypass,data);
	if(index==0){
		// printf("\nstart_addr:%lx total_length:%d\n",start_addr,total_length);
		// printf("each_client_length : %d\n",each_client_length);
		// printf("addr_start_64B_low : %x\n",addr_start_64B_low);
		// printf("addr_start_64B_high : %x\n",addr_start_64B_high);
		// printf("head_length : %d\n",head_length);
		// printf("tail_length : %d\n",tail_length);
		size_t timeout=clock64();
		while(*reg_response != (last_response+1)){
			if(clock64()-timeout>size_t(3)*3000000000){
				printf("Error:\n");
				printf("Count:%ld\n",count);
				printf("reg_rx_overflow:%d\n",*reg_rx_overflow);
				while(1);
			}
			// printf("reg_response:%d\n",*reg_response);
			// cu_sleep(1);
		}
		count+=1;
		// printf("reg_response:%d\n",*reg_response);

		// for(int i=0;i<128;i++){
		// 	if(i%16==0){
		// 		printf("\n");
		// 	}
		// 	printf("%x ",aligned_start[i]);
		// }
		// printf("\n\n\n======================================\n");
		// cu_sleep(1);
		// if(size>128){
		// 	for(int i=0;i<16;i++){
		// 		printf("%f ",start_addr[i]);
		// 	}
		// 	printf("\n");
		// 	for(int i=size/4-16;i<size/4;i++){
		// 		printf("%f ",start_addr[i]);
		// 	}
		// 	printf("\n");
		// }else{
		// 	for(int i=0;i<size/4;i++){
		// 		printf("%f ",start_addr[i]);
		// 		// start_addr[i]+=10;
		// 	}
		// 	printf("\n");
		// }
		
	}
	__syncthreads();
	

	
	// for(int i=0;i<size/4;i++){
	// 	printf("%f ",start_addr[i]);
	// 	// start_addr[i]+=10;
	// }
}

void cu_all_reduce_float(float* start_addr, int size, unsigned int ** fpga_regs, unsigned int *params, cudaStream_t stream){
	
	unsigned int * reg_response			=	fpga_regs[0];
	unsigned int * reg_bypass			=	fpga_regs[1];
	unsigned int * reg_rx_overflow		=	fpga_regs[2];


	unsigned int total_length			=	params[0];
	unsigned int each_client_length		=	params[1];
	unsigned int addr_start_64B_low		=	params[2];
	unsigned int addr_start_64B_high	=	params[3];
	unsigned int head_length			=	params[4];
	unsigned int tail_length			=	params[5];
	unsigned int dtype 					=	params[6];

		kernel_all_reduce_float<<<1,16,0,stream>>>(	start_addr,
												size,
												reg_bypass,
												reg_response,

												total_length,
												each_client_length,
												addr_start_64B_low,
												addr_start_64B_high,
												head_length,
												tail_length,
												dtype,
												reg_rx_overflow
												);
		// cudaDeviceSynchronize();
	
}

void cu_all_reduce_float_gloo(float* start_addr, int size, unsigned int ** fpga_regs, unsigned int *params){
	unsigned int * reg_response			=	fpga_regs[0];
	unsigned int * reg_bypass			=	fpga_regs[1];

	unsigned int total_length			=	params[0];
	unsigned int each_client_length		=	params[1];
	unsigned int addr_start_64B_low		=	params[2];
	unsigned int addr_start_64B_high	=	params[3];
	unsigned int head_length			=	params[4];
	unsigned int tail_length			=	params[5];

	// kernel_all_reduce_float<<<1,16>>>(			start_addr,
	// 											size,
	// 											reg_bypass,
	// 											reg_response,

	// 											total_length,
	// 											each_client_length,
	// 											addr_start_64B_low,
	// 											addr_start_64B_high,
	// 											head_length,
	// 											tail_length,
	// 											0
	// 											);
}