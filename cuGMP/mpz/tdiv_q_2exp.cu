#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/helper_cuda.h"

#include "../cuGMP.h"


__global__ void shiftRightKernel(__dev_mpz_struct *res, __dev_mpz_struct *a, unsigned int shift_limbs, unsigned int shift_bits, unsigned int result_limbs)
{
	unsigned int i = THREAD_ID;

	if (i < result_limbs)
	{
		if (i < result_limbs - 1)
		{
			res->_mp_d[i] = (a->_mp_d[i + shift_limbs] >> shift_bits) + (a->_mp_d[i + shift_limbs + 1] << (GMP_LIMB_BITS - shift_bits));
		}
		else
		{
			res->_mp_d[i] = (a->_mp_d[i + shift_limbs] >> shift_bits);
		}
	}
}

void mpz_tdiv_q_2exp(mpz_ptr res, mpz_ptr a, unsigned long int exponent)
{
	unsigned int result_limbs = ABS(a->_mp_size) - exponent / GMP_LIMB_BITS;
	allocate_memory(res, result_limbs, result_limbs);

	unsigned int shift_limbs = exponent / GMP_LIMB_BITS;
	unsigned int shift_bits = exponent % GMP_LIMB_BITS;

#ifdef KERNEL_PRINT
	printf("shiftRightKernel <<<%d, %d>>>\n", result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	shiftRightKernel << <result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(res->_dev_mp_struct, a->_dev_mp_struct, shift_limbs, shift_bits, result_limbs);
	getLastCudaError("Kernel execution failed: [ shiftRightKernel ]");
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
