#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/helper_cuda.h"

#include "../cuGMP.h"

#define BORROW 0x01
#define PROPAGATE 0xFF

// Should operands be swapped?
__device__ bool d_SwapOperands;

// Kernel that is responsible for evaluation of operand order
__global__ void subSwapOperandsKernel(const __dev_mpz_struct *a, const __dev_mpz_struct *b)
{
	int i = a->_mp_size - 1;

	if (b->_mp_size > a->_mp_size)
	{
		d_SwapOperands = true;
		return;
	}
	if (a->_mp_size > b->_mp_size)
	{
		d_SwapOperands = false;
		return;
	}

	while (a->_mp_d[i] == b->_mp_d[i] && i > 0)
	{
		i--;
	}

	if (a->_mp_d[i] >= b->_mp_d[i])
	{
		d_SwapOperands = false;
	}
	else
	{
		d_SwapOperands = true;
	}
}

// b is always the smaller number
__global__ void subKernel(__dev_mpz_struct *res, const __dev_mpz_struct *a, const __dev_mpz_struct *b, unsigned char *borrow)
{
	int i = THREAD_ID;

	if (i >= res->_mp_alloc)
	{
		return;
	}

	// Sub only existing limbs
	if (ABS(a->_mp_size) > i && ABS(b->_mp_size) > i)
	{
		res->_mp_d[i] = a->_mp_d[i] - b->_mp_d[i];
		borrow[i] = (res->_mp_d[i] > a->_mp_d[i]);
	}
	else
	{
		borrow[i] = 0;
		if (ABS(a->_mp_size) > i)
		{
			res->_mp_d[i] = a->_mp_d[i];
		}
		else
		{
			res->_mp_d[i] = 0;
		}
	}

	if (res->_mp_d[i] == 0)
	{
		borrow[i] = PROPAGATE;
	}
}

// Split in two kernels, because overflow can be propagated only after all limb additions have been completed (full synchronization)
__global__ void borrowPropagateKernel(__dev_mpz_struct *res, unsigned char *borrow)
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
		if (borrow[i] == 0)
		{
			break;
		}

		if (borrow[i] == BORROW)
		{
			res->_mp_d[dest]--;
			break;
		}

		// if (borrow[i] == PROPAGATE) continue;
	}
}

// Size of result number may increase - detect this and adjust size.
// Run only on one thread - HOST does not have access to CUDA memory, so it has to be a kernel.
__global__ void calculateSubResultSizeKernel(__dev_mpz_struct *res)
{
	int i = THREAD_ID;

	if (i < res->_mp_alloc - 1)
	{
		if (res->_mp_d[i + 1] == 0 && res->_mp_d[i] > 0)
		{
#if __CUDA_ARCH__ >= 200
			atomicMax(&res->_mp_size, i + 1);
#endif
		}
	}
}

void mpz_sub(mpz_ptr res, mpz_ptr a, mpz_ptr b)
{
	// Carry buffer.
	unsigned char *dev_carry = 0;
	bool swapOperands = false;

	// Determine result size + allocate mem.
	int size = MAX(ABS(a->_mp_size), ABS(b->_mp_size)) + 1;
	// Max result size is "size", but minimum result size is "size-1" - this is calculated in calculateResultSizeKernel
	allocate_memory(res, size, size - 1);

	// Allocate carry buffer (could be eliminated)
	checkCudaErrors(cudaMalloc((void**)&dev_carry, res->_mp_alloc * sizeof(unsigned char)));

#ifdef KERNEL_PRINT
	printf("subSwapOperandsKernel <<<%d, %d>>>\n", 1, 1);
#endif
	subSwapOperandsKernel << <1, 1 >> >(a->_dev_mp_struct, b->_dev_mp_struct);
	getLastCudaError("Kernel execution failed: [ subSwapOperandsKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	cudaMemcpyFromSymbol(&swapOperands, d_SwapOperands, sizeof(swapOperands), 0, cudaMemcpyDeviceToHost);

	if (swapOperands)
	{
		mpz_ptr tmp = a;
		a = b;
		b = tmp;
	}

#ifdef KERNEL_PRINT
	printf("subKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	subKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct, a->_dev_mp_struct, b->_dev_mp_struct, dev_carry);
	getLastCudaError("Kernel execution failed: [ subKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

#ifdef KERNEL_PRINT
	printf("borrowPropagateKernel <<<%d, %d>>>\n", size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	borrowPropagateKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct, dev_carry);
	getLastCudaError("Kernel execution failed: [ borrowPropagateKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	// Calculate result size.
#ifdef KERNEL_PRINT
	printf("calculateSubResultSizeKernel <<<%d, %d>>>\n", 1, 1);
#endif
	calculateSubResultSizeKernel << <size / BLOCK_SIZE + 1, BLOCK_SIZE >> >(res->_dev_mp_struct);
	getLastCudaError("Kernel execution failed: [ calculateSubResultSizeKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	// Extract result size.
	copy_operand_data_without_limbs(res, MemcpyDirection::memcpyDeviceToHost);

	if (swapOperands)
	{
		res->_mp_size = -1 * res->_mp_size;
		copy_operand_data_without_limbs(res, MemcpyDirection::memcpyHostToDevice);
	}

	cudaFree(dev_carry);
}