#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/helper_cuda.h"

#include "../cuGMP.h"

__global__ void orKernel(__dev_mpz_struct *res, const __dev_mpz_struct *a, const __dev_mpz_struct *b, unsigned int size)
{
	unsigned int i = THREAD_ID;

	if (i < size)
	{
		if (i >= a->_mp_size)
		{
			res->_mp_d[i] = b->_mp_d[i];
		}
		else if (i >= b->_mp_size)
		{
			res->_mp_d[i] = a->_mp_d[i];
		}
		else
		{
			res->_mp_d[i] = a->_mp_d[i] | b->_mp_d[i];
		}
	}
}

void mpz_ior(mpz_ptr res, mpz_ptr a, mpz_ptr b)
{
	unsigned int size = MAX(ABS(a->_mp_size), ABS(b->_mp_size));
	allocate_memory(res, size, size);

#ifdef KERNEL_PRINT
	printf("orKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	orKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct, a->_dev_mp_struct, b->_dev_mp_struct, size);
	getLastCudaError("Kernel execution failed: [ orKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}
