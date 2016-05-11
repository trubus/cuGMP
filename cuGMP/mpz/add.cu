#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/helper_cuda.h"

#include "../cuGMP.h"

#define CARRY 0x01
#define PROPAGATE 0xFF

__global__ void addKernel(__dev_mpz_struct *res, const __dev_mpz_struct *a, const __dev_mpz_struct *b, unsigned char *carry)
{
	int i = THREAD_ID;

	if (i >= res->_mp_alloc)
	{
		return;
	}

	// Sum only existing limbs
	if (ABS(a->_mp_size) > i && ABS(b->_mp_size) > i)
	{
		res->_mp_d[i] = a->_mp_d[i] + b->_mp_d[i];
		carry[i] = (res->_mp_d[i] < a->_mp_d[i]);
	}
	else
	{
		carry[i] = 0;
		if (ABS(a->_mp_size) > i)
		{
			res->_mp_d[i] = a->_mp_d[i];
		}
		else if (ABS(b->_mp_size) > i)
		{
			res->_mp_d[i] = b->_mp_d[i];
		}
		else
		{
			res->_mp_d[i] = 0;
		}
	}
	
	if (res->_mp_d[i] == -1llu)
	{
		carry[i] = PROPAGATE;
	}
}

// Split in two kernels, because overflow can be propagated only after all limb additions have been completed (full synchronization)
__global__ void carryPropagateKernel(__dev_mpz_struct *res, unsigned char *carry)
{
	int i = THREAD_ID;
	int dest = i;

	if (i >= res->_mp_alloc)
	{
		return;
	}

	while (i > 0)
	{
		i--;
		if (carry[i] == 0)
		{
			break;
		}

		if (carry[i] == CARRY)
		{
			res->_mp_d[dest]++;
			break;
		}

		// if (carry[i] == PROPAGATE) continue;
	}
}

// Size of result number may increase - detect this and adjust size.
// Run only on one thread - HOST does not have access to CUDA memory, so it has to be a kernel.
__global__ void calculateAddResultSizeKernel(__dev_mpz_struct *res)
{
	if (res->_mp_d[ABS(res->_mp_size)] > 0)
	{
		if (res->_mp_size < 0)
		{
			res->_mp_size--;
		}
		else
		{
			res->_mp_size++;
		}
	}
}

void mpz_add(mpz_ptr res, mpz_ptr a, mpz_ptr b)
{
	// Carry buffer.
	unsigned char *dev_carry = 0;

	// Determine result size + allocate mem.
	int size = MAX(ABS(a->_mp_size), ABS(b->_mp_size)) + 1;
	// Max result size is "size", but minimum result size is "size-1" - this is calculated in calculateResultSizeKernel
	allocate_memory(res, size, size - 1);

	// Allocate carry buffer (could be eliminated)
	checkCudaErrors(cudaMalloc((void**)&dev_carry, res->_mp_alloc * sizeof(unsigned char)));

#ifdef KERNEL_PRINT
	printf("addKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	addKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct, a->_dev_mp_struct, b->_dev_mp_struct, dev_carry);
	getLastCudaError("Kernel execution failed: [ addKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

#ifdef KERNEL_PRINT
	printf("carryPropagateKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	carryPropagateKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct, dev_carry);
	getLastCudaError("Kernel execution failed: [ carryPropagateKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	// Calculate result size.
#ifdef KERNEL_PRINT
	printf("calculateAddResultSizeKernel <<<%d, %d>>>\n", 1, 1);
#endif
	calculateAddResultSizeKernel << <1, 1 >> >(res->_dev_mp_struct);
	getLastCudaError("Kernel execution failed: [ calculateAddResultSize ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	// Extract result size.
	copy_operand_data_without_limbs(res, MemcpyDirection::memcpyDeviceToHost);

	cudaFree(dev_carry);
}