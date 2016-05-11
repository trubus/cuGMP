#include <stdio.h>

#include "../cuGMP.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/helper_cuda.h"

void init(mpz_ptr x, int limbs)
{
	__dev_mpz_struct dev_x;
	dev_x._mp_alloc = limbs;
	dev_x._mp_size = 0;
	x->_mp_alloc = limbs;
	x->_mp_size = 0;

	checkCudaErrors(cudaMalloc((void**)&dev_x._mp_d, sizeof(mp_limb_t) * limbs));
	x->_mp_d = (mp_limb_t *)malloc(limbs * sizeof(x->_mp_d));

	checkCudaErrors(cudaMalloc((void**)&x->_dev_mp_struct, sizeof(__dev_mpz_struct)));
	checkCudaErrors(cudaMemcpy(x->_dev_mp_struct, &dev_x, sizeof(__dev_mpz_struct), cudaMemcpyHostToDevice));
}

void mpz_init(mpz_ptr x)
{
	init(x, 1);
}


void mpz_init2(mpz_ptr x, mp_bitcnt_t bits)
{
	int limbs;
	limbs = (int) ((bits + GMP_LIMB_BITS - 1) / GMP_LIMB_BITS);
	limbs = MAX(limbs, 1);
	
	init(x, limbs);
}


void mpz_clear(mpz_ptr x)
{
	__dev_mpz_struct dev_x;
	checkCudaErrors(cudaMemcpy(&dev_x, x->_dev_mp_struct, sizeof(__dev_mpz_struct), cudaMemcpyDeviceToHost));
	free(x->_mp_d);
	cudaFree(dev_x._mp_d);
	cudaFree(x->_dev_mp_struct);
}

void printDevices(void)
{
	int nDevices = 0;
	size_t free, total;

	checkCudaErrors(cudaMemGetInfo(&free, &total));
#ifdef _WIN32
	printf("GPU0 memory %Iu / %IuMB (free/total)\n\n", free / 1048576, total / 1048576);
#else
	printf("GPU0 memory %zu / %zuMB (free/total)\n\n", free / 1048576, total / 1048576);
#endif

	checkCudaErrors(cudaGetDeviceCount(&nDevices));

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
}

void cudaInit(void)
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
}

void cudaReset(void)
{
	cudaDeviceReset();
}
