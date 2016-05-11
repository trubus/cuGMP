#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/helper_cuda.h"

#include "../cuGMP.h"


__global__ void shiftLeftKernel(__dev_mpz_struct *res, __dev_mpz_struct *a, unsigned int shift_limbs, unsigned int shift_bits, unsigned int source_limbs)
{
	unsigned int i = THREAD_ID;

	if (i <= source_limbs)
	{
		if (i < shift_limbs)
		{
			res->_mp_d[i] = 0;
		}
		else if (i == shift_limbs)
		{
			res->_mp_d[i] = (a->_mp_d[i - shift_limbs] << shift_bits);
		}
		else if (i < source_limbs)
		{
			res->_mp_d[i] = (a->_mp_d[i - shift_limbs] << shift_bits) + (a->_mp_d[i - shift_limbs - 1] >> (GMP_LIMB_BITS - shift_bits));
		}
		else if (i == source_limbs)
		{
			res->_mp_d[i] = (a->_mp_d[i - shift_limbs - 1] >> (GMP_LIMB_BITS - shift_bits));
			if (res->_mp_d[i] > 0)
			{
				res->_mp_size = i + 1;
			}
		}
	}
}

void mpz_mul_2exp(mpz_ptr res, mpz_ptr a, unsigned long int exponent)
{
	unsigned int source_limbs = ABS(a->_mp_size) + exponent / GMP_LIMB_BITS;
	unsigned int result_limbs = source_limbs + 1;
	allocate_memory(res, result_limbs, source_limbs);

	unsigned int shift_limbs = exponent / GMP_LIMB_BITS;
	unsigned int shift_bits = exponent % GMP_LIMB_BITS;

#ifdef KERNEL_PRINT
	printf("shiftLeftKernel <<<%d, %d>>>\n", result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	shiftLeftKernel << <result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(res->_dev_mp_struct, a->_dev_mp_struct, shift_limbs, shift_bits, source_limbs);
	getLastCudaError("Kernel execution failed: [ shiftLeftKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	// mp_size could have changed on the device - reflect on the host.
	copy_operand_data_without_limbs(res, MemcpyDirection::memcpyDeviceToHost);

	// If the number shifted is negative, reflect it in the result.
	if (a->_mp_size < 0)
	{
		res->_mp_size = -1 * ABS(res->_mp_size);
		copy_operand_data_without_limbs(res, MemcpyDirection::memcpyHostToDevice);
	}
}
