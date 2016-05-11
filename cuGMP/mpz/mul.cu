#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

#include "../include/helper_cuda.h"

#include "../cuGMP.h"

#define MUL_BASE_BITS 16
#define MUL_BASE_SIZE_COEF (GMP_LIMB_BITS / MUL_BASE_BITS)
#define MUL_BASE_MASK (((uint64_t)1 << MUL_BASE_BITS) - 1)

//#define MUL_DEBUG


__global__ void zeroBuffersKernel(cufftDoubleComplex *dev_arr_res, cufftDoubleComplex *dev_arr_a, cufftDoubleComplex *dev_arr_b, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
		dev_arr_a[i].x = 0;
		dev_arr_a[i].y = 0;
		
		dev_arr_b[i].x = 0;
		dev_arr_b[i].y = 0;
		
		dev_arr_res[i].x = 0;
		dev_arr_res[i].y = 0;
	}
}

__global__ void polynomialRepresentationKernel(__dev_mpz_struct *a, __dev_mpz_struct *b, cufftDoubleComplex *dev_arr_a, cufftDoubleComplex *dev_arr_b, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
#ifdef MUL_DEBUG
		printf("[%d] a = %llu, b = %llu\n", i, (uint64_t)a->_mp_d[i], (uint64_t)b->_mp_d[i]);
#endif

		for (int j = 0; j < MUL_BASE_SIZE_COEF; j++)
		{
			int target = (i * MUL_BASE_SIZE_COEF) + j;
			
			// Extract MUL_BASE_BITS from source to target variable
			if (i < ABS(a->_mp_size))
			{
				dev_arr_a[target].x = (a->_mp_d[i] & (MUL_BASE_MASK << (j * MUL_BASE_BITS))) >> (j * MUL_BASE_BITS);
				dev_arr_a[target].y = 0;
			}

			if (i < ABS(b->_mp_size))
			{
				dev_arr_b[target].x = (b->_mp_d[i] & (MUL_BASE_MASK << (j * MUL_BASE_BITS))) >> (j * MUL_BASE_BITS);
				dev_arr_b[target].y = 0;
			}

#ifdef MUL_DEBUG
			printf("dev_arr_a[%d] x = %.1f, y = %.1f, dev_arr_b[%d] x = %.1f, y = %.1f\n",
				target,
				dev_arr_a[target].x,
				dev_arr_a[target].y,
				target,
				dev_arr_b[target].x,
				dev_arr_b[target].y);
#endif
		}
	}
}

__global__ void pointwiseMultiplicationKernel(cufftDoubleComplex *c, cufftDoubleComplex *a, cufftDoubleComplex *b, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
		// Source: 3_big.pdf VI. 6.
		c[i].x = a[i].x * b[i].x - a[i].y * b[i].y;
		c[i].y = a[i].x * b[i].y + a[i].y * b[i].x;
#ifdef MUL_DEBUG
		printf("[%d] (%.1f + %.1fi) * (%.1f + %.1fi)  = (%.1f + %.1fi)\n", i,
			a[i].x, a[i].y,
			b[i].x, b[i].y,
			c[i].x, c[i].y);
#endif
	}
}

__global__ void resultNormalisationKernel(cufftDoubleComplex *c, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
#ifdef MUL_DEBUG
		double tmp = c[i].x;
#endif
		c[i].x = c[i].x / size;
		// The y value is not used in extractResultKernel.
		//c[i].y = c[i].y / size;
#ifdef MUL_DEBUG
		if (c[i].x * size != tmp)
		{
			printf("\t[%d] %f != %f\n", i, c[i].x, tmp);
		}

		printf("c[%d] = (%.1f + %.1fi) size %d\n", i, c[i].x, c[i].y, size);

		if (c[i].y > 0.001 || c[i].y < -0.001)
		{
			printf("result imaginary non-zero c[%d] = (%.1f + %.1fi)\n", i, c[i].x, c[i].y);
		}

		if (c[i].x > MUL_BASE_MASK)
		{
			// If some number is greater than X print it (X is determined by MUL_BASE_BITS...) - detect overflow.
			printf("result overflows c[%d] = (%.1f + %.1fi) (%llu)\n", i, c[i].x, c[i].y, MUL_BASE_MASK);
		}
#endif
	}
}

// Reverse process of polynomialRepresentationKernel
__global__ void extractResultKernel(__dev_mpz_struct *res_lsb, __dev_mpz_struct *res_msb, cufftDoubleComplex *c, unsigned int size)
{
	int i = THREAD_ID;

	if (i < size)
	{
		res_lsb->_mp_d[i] = 0;
		res_msb->_mp_d[i] = 0;

		for (int j = 0; j < MUL_BASE_SIZE_COEF; j++)
		{
			uint64_t source = llrint(c[i * MUL_BASE_SIZE_COEF + j].x);

			// If source is less than zero, assigning it to the result overflows, therefore results in large error
			// This occurs mainly for MUL_BASE_BITS 32 where errors are almost guaranteed to occur.
			if (c[i * MUL_BASE_SIZE_COEF + j].x < 0)
			{
#ifdef MUL_DEBUG
				printf("source less than zero detected %f\n", c[i * MUL_BASE_SIZE_COEF + j].x);
#endif
				source = 0;
			}

			res_lsb->_mp_d[i] += (source & MUL_BASE_MASK) << (j * MUL_BASE_BITS);
			res_msb->_mp_d[i] += ((source & (MUL_BASE_MASK << MUL_BASE_BITS)) >> MUL_BASE_BITS) << (j * MUL_BASE_BITS);
		}

#ifdef MUL_DEBUG
		printf("res_lsb->_mp_d[%d] = %llu res_msb->_mp_d[%d] = %llu sum = %llu\n", i,
			(uint64_t)res_lsb->_mp_d[i], i,
			(uint64_t)res_msb->_mp_d[i],
			((uint64_t)res_msb->_mp_d[i] << MUL_BASE_BITS) + res_lsb->_mp_d[i]);
#endif

		// Calculate size
		// Possible to rewrite using atomicMax similarly as in mpz_sub
		if (i == size - 1)
		{
			unsigned int j = i;
			while (res_lsb->_mp_d[j] == 0 && j > 0)
			{
				j--;
			}
			res_lsb->_mp_size = j + 1;

			j = i;
			while (res_msb->_mp_d[j] == 0 && j > 0)
			{
				j--;
			}
			res_msb->_mp_size = j + 1;
		}
	}
}

// First call to cufftPlan takes a lot of time, every other call is quick - this heats up CUDA for accurate measurement
void cuFFT_init()
{
	cufftHandle plan;
	cufftPlan1d(&plan, 1024, CUFFT_Z2Z, 1);
	cufftDestroy(plan);
}

void mpz_mul(mpz_ptr res, mpz_ptr a, mpz_ptr b)
{
	// Device multiplication buffers
	cufftDoubleComplex *dev_arr_a = 0;
	cufftDoubleComplex *dev_arr_b = 0;
	cufftDoubleComplex *dev_arr_res = 0;

	// Determine result result_limbs + allocate mem.
	unsigned int result_limbs = 2 * MAX(ABS(a->_mp_size), ABS(b->_mp_size));
	unsigned int cuda_fft_complex_limbs = result_limbs * MUL_BASE_SIZE_COEF;

	unsigned int cufft_transform_size = 1;
	while (cufft_transform_size < cuda_fft_complex_limbs)
		cufft_transform_size = cufft_transform_size << 1;

#ifdef MUL_DEBUG
	printf("result_limbs=%d cuda_fft_complex_limbs=%d cufft_transform_size=%d\n", result_limbs, cuda_fft_complex_limbs, cufft_transform_size);
#endif
	allocate_memory(res, result_limbs, result_limbs);

	// Partial results - least and most significant bits.
	mpz_t res_lsb;
	mpz_t res_msb_tmp;
	mpz_t res_msb;
	mpz_init2(res_lsb, result_limbs * GMP_LIMB_BITS);
	mpz_init2(res_msb_tmp, result_limbs * GMP_LIMB_BITS);
	mpz_init2(res_msb, result_limbs * GMP_LIMB_BITS);

	// Allocate multiplication buffers
	checkCudaErrors(cudaMalloc((void**)&dev_arr_a, sizeof(cufftDoubleComplex) * cufft_transform_size));
	checkCudaErrors(cudaMalloc((void**)&dev_arr_b, sizeof(cufftDoubleComplex) * cufft_transform_size));
	checkCudaErrors(cudaMalloc((void**)&dev_arr_res, sizeof(cufftDoubleComplex) * cufft_transform_size));

#ifdef KERNEL_PRINT
	printf("zeroBuffersKernel <<<%d, %d>>>\n", cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	zeroBuffersKernel << <cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(dev_arr_res, dev_arr_a, dev_arr_b, cufft_transform_size);
	getLastCudaError("Kernel execution failed: [ zeroBuffersKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

#ifdef KERNEL_PRINT
	printf("polynomialRepresentationKernel <<<%d, %d>>>\n", result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	polynomialRepresentationKernel << <result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(a->_dev_mp_struct, b->_dev_mp_struct, dev_arr_a, dev_arr_b, result_limbs);
	getLastCudaError("Kernel execution failed: [ polynomialRepresentation ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

#ifdef MUL_DEBUG
	printf("Preparing CUFFT plan(%d)\n", cufft_transform_size);
#endif
	cufftHandle plan;
	cufftPlan1d(&plan, cufft_transform_size, CUFFT_Z2Z, 1);

	// Transform operands from time to frequency domain.
#ifdef MUL_DEBUG
	printf("Transforming signal cufftExecC2C\n");
#endif
	cufftExecZ2Z(plan, dev_arr_a, dev_arr_a, CUFFT_FORWARD);
	getLastCudaError("cuFFT execution failed: [ dev_arr_a, dev_arr_a, CUFFT_FORWARD ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	cufftExecZ2Z(plan, dev_arr_b, dev_arr_b, CUFFT_FORWARD);
	getLastCudaError("cuFFT execution failed: [ dev_arr_b, dev_arr_b, CUFFT_FORWARD ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif
#ifdef MUL_DEBUG
	printf("cuFFT FORWARD finished\n");
#endif

#ifdef KERNEL_PRINT
	printf("pointwiseMultiplicationKernel <<<%d, %d>>>\n", cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	pointwiseMultiplicationKernel << <cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(dev_arr_res, dev_arr_a, dev_arr_b, cufft_transform_size);
	getLastCudaError("Kernel execution failed: [ pointwiseMultiplicationKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	cufftExecZ2Z(plan, dev_arr_res, dev_arr_res, CUFFT_INVERSE);
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif
#ifdef MUL_DEBUG
	printf("cuFFT INVERSE finished\n");
#endif

#ifdef KERNEL_PRINT
	printf("resultNormalisationKernel <<<%d, %d>>>\n", cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	resultNormalisationKernel << <cufft_transform_size / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(dev_arr_res, cufft_transform_size);
	getLastCudaError("Kernel execution failed: [ resultNormalisationKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

#ifdef KERNEL_PRINT
	printf("extractResultKernel <<<%d, %d>>>\n", result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE);
#endif
	extractResultKernel << <result_limbs / BLOCK_SIZE + 1, BLOCK_SIZE >> >
		(res_lsb->_dev_mp_struct, res_msb_tmp->_dev_mp_struct, dev_arr_res, result_limbs);
	getLastCudaError("Kernel execution failed: [ extractResultKernel ]");
#ifdef EXPLICIT_SYNCHRONIZATION
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	copy_operand_data_without_limbs(res_lsb, memcpyDeviceToHost);
	copy_operand_data_without_limbs(res_msb_tmp, memcpyDeviceToHost);

	mpz_mul_2exp(res_msb, res_msb_tmp, MUL_BASE_BITS);
#ifdef MUL_DEBUG
	printOperandCuda(res_msb_tmp);
	printOperandCuda(res_msb);
#endif

	mpz_add(res, res_msb, res_lsb);

#ifdef MUL_DEBUG
	printOperandCuda(res);
#endif

	// Free device multiplication buffers
	cudaFree(dev_arr_a);
	cudaFree(dev_arr_b);
	cudaFree(dev_arr_res);

	cufftDestroy(plan);

	mpz_clear(res_lsb);
	mpz_clear(res_msb_tmp);
	mpz_clear(res_msb);
}