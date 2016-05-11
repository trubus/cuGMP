#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/helper_cuda.h"

#include "../cuGMP.h"

__global__ void printOperandKernel(__dev_mpz_struct *x, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
		// Beware: Cuda does support limited printing:
		// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
		printf("[%d] %llX = %llu\n",
			i,
			(uint64_t)x->_mp_d[i],
			(uint64_t)x->_mp_d[i]);
	}
}


void printOperandCuda(mpz_ptr x)
{
	// Launch a kernel on the GPU with one thread for each element.
	unsigned int size = ABS(x->_mp_size);

#ifdef KERNEL_PRINT
	printf("printOperandKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	printOperandKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE>> >(x->_dev_mp_struct, size);
	getLastCudaError("Kernel execution failed: [ printOperandKernel ]");

#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}