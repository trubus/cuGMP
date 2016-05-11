#include <stdio.h>

#include "cuGMP.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/helper_cuda.h"

void allocate_memory(mpz_ptr x, size_t limbs, size_t size)
{
	__dev_mpz_struct dev_x;
	checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));

	if (x->_mp_alloc < limbs)
	{
		cudaFree(dev_x._mp_d);
		free(x->_mp_d);
		checkCudaErrors(cudaMalloc((void**)&dev_x._mp_d, sizeof(mp_limb_t) * limbs));
		x->_mp_d = (mp_limb_t *)malloc(limbs * sizeof(x->_mp_d));
		x->_mp_alloc = (int)limbs;
	}
	x->_mp_size = (int)size;

	dev_x._mp_size = x->_mp_size;
	dev_x._mp_alloc = x->_mp_alloc;
	checkCudaErrors(cudaMemcpy(x->_dev_mp_struct, &dev_x, sizeof(__dev_mpz_struct), cudaMemcpyHostToDevice));
}

__dev_mpz_struct copy_operand(mpz_ptr x)
{
	__dev_mpz_struct dev_x;
	checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
	return dev_x;
}

void copy_operand(mpz_ptr x, __dev_mpz_struct dev_x)
{
	checkCudaErrors(cudaMemcpy(x->_dev_mp_struct, &dev_x, sizeof(__dev_mpz_struct), cudaMemcpyHostToDevice));
}

void copy_operand_data_without_limbs(mpz_ptr x, MemcpyDirection direction)
{
	__dev_mpz_struct dev_x;

	switch (direction)
	{
	case MemcpyDirection::memcpyHostToDevice:
		// To preserve device pointers, it is vital to first copy actual struct from device, before altering it.
		checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
		dev_x._mp_size = x->_mp_size;
		checkCudaErrors(cudaMemcpy(x->_dev_mp_struct, &dev_x, sizeof(__dev_mpz_struct), cudaMemcpyHostToDevice));
		break;
	case MemcpyDirection::memcpyDeviceToHost:
		checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
		x->_mp_size = dev_x._mp_size;
		break;
	default:
		printf("Unsupported direction in copy_operand_data, skipping\n");
		break;
	}
}

void copy_operand_data_with_limbs(mpz_ptr x, MemcpyDirection direction)
{
	__dev_mpz_struct dev_x;

	switch (direction)
	{
	case MemcpyDirection::memcpyHostToDevice:
		checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(dev_x._mp_d, x->_mp_d, sizeof(mp_limb_t) * ABS(x->_mp_size), cudaMemcpyHostToDevice));
		break;
	case MemcpyDirection::memcpyDeviceToHost:
		checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(x->_mp_d, dev_x._mp_d, sizeof(mp_limb_t) * ABS(dev_x._mp_size), cudaMemcpyDeviceToHost));
		break;
	default:
		printf("Unsupported direction in copy_operand_data, skipping\n");
		break;
	}
}
