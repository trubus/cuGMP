#include "stdafx.h"

#include "TestCuGMP.h"

#include "Measurement.h"

#include <cuGMP.h>

#include "Tests.h"

#ifndef _WIN32
#include <cstring>
#endif

//#define CUDA_DEBUG

void Init()
{
#ifdef CUDA_DEBUG
	printf("CUDA and cuFFT init");
#endif
	cudaInit();
	cuFFT_init();
#ifdef CUDA_DEBUG
	printf("\t\t[FINISHED].\n");
#endif
}

void PrintDevices(void)
{
	printDevices();
}

uint64_t CUDAAddition(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	// Variables
	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	//mpz_init(c);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

#ifdef CUDA_DEBUG
	printf("printOperandCuda(a)\n");
	printOperandCuda(a);
	printf("printOperandCuda(b)\n");
	printOperandCuda(b);
#endif

	start_measurement(&measurement);
	for (size_t i = 0; i < ADD_ITERATIONS; i++)
	{
		mpz_add(c, a, b);
	}
	stop_measurement(&measurement);

#ifdef CUDA_DEBUG
	printf("printOperandCuda(result)\n");
	printOperandCuda(c);

	char * result_hex = mpz_get_str(NULL, 16, c);
	printf("printing mpz_get_str_hex\n%s\n", result_hex);
	free(result_hex);
#endif

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, ADD_ITERATIONS);
}

uint64_t CUDASubtraction(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	// Variables
	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	//mpz_init(c);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

#ifdef CUDA_DEBUG
	printf("printOperandCuda(a)\n");
	printOperandCuda(a);
	printf("printOperandCuda(b)\n");
	printOperandCuda(b);
#endif

	start_measurement(&measurement);
	for (size_t i = 0; i < SUB_ITERATIONS; i++)
	{
		mpz_sub(c, a, b);
	}
	stop_measurement(&measurement);

#ifdef CUDA_DEBUG
	printf("printOperandCuda(result)\n");
	printOperandCuda(c);

	char * result_hex = mpz_get_str(NULL, 16, c);
	printf("printing mpz_get_str_hex\n%s\n", result_hex);
	free(result_hex);
#endif

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, SUB_ITERATIONS);
}

uint64_t CUDAMultiplication(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 2 * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < MUL_ITERATIONS; i++)
	{
		mpz_mul(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, MUL_ITERATIONS);
}

uint64_t CUDABitShiftLeft(char * result, const char * hex_a, unsigned long shift_bits)
{
	Init();

	__measurement measurement;

	mpz_t a, c;

	mpz_init(a);
	mpz_init2(c, strlen(hex_a) * 4 + shift_bits + GMP_LIMB_BITS);

	mpz_set_str(a, hex_a, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < SHIFT_ITERATIONS; i++)
	{
		mpz_mul_2exp(c, a, shift_bits);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, SHIFT_ITERATIONS);
}

uint64_t CUDABitShiftRight(char * result, const char * hex_a, unsigned long shift_bits)
{
	Init();

	__measurement measurement;

	mpz_t a, c;

	mpz_init(a);
	mpz_init2(c, strlen(hex_a) * 4 - shift_bits + GMP_LIMB_BITS);

	mpz_set_str(a, hex_a, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < SHIFT_ITERATIONS; i++)
	{
		mpz_tdiv_q_2exp(c, a, shift_bits);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, SHIFT_ITERATIONS);
}

uint64_t CUDAAnd(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_and(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}

uint64_t CUDAOr(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_ior(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}

uint64_t CUDAXor(char * result, const char * hex_a, const char * hex_b)
{
	Init();

	__measurement measurement;

	mpz_t a, b;
	mpz_t c;

	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, MAX(strlen(hex_a), strlen(hex_b)) * 4 + 64);

	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_xor(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result, 16, c);

	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	cudaReset();

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}


// Testing purposes
void TestCuGMP(char * result, const char * hex_a, const char * hex_b)
{
	//size_t len = 2 * MAX(strlen(hex_a), strlen(hex_b)) + 16;
	//char * result = (char *)malloc(len * sizeof(char));

	CUDAMultiplication(result, hex_a, hex_b);

	/*printf("mpz_get_str(a): %s\n", hex_a);
	printf("mpz_get_str(b): %s\n", hex_b);*/
	printf("mpz_get_str(result): %s\n", result);

	//free(result);
}
