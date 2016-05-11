#include "TestGMP.h"

#ifndef _WIN32
#include <cstdlib>
#include <cstddef>
#endif

#include <time.h>
#include <gmp.h>
#include "Tests.h"
#include "Measurement.h"

uint64_t GMPRandomAddition(testResult result, const uint64_t bits, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits + 64);

	mpz_urandomb(a, state, bits);
	mpz_urandomb(b, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < ADD_ITERATIONS; i++)
	{
		mpz_add(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	// Free memory
	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, ADD_ITERATIONS);
}

uint64_t GMPRandomSubtraction(testResult result, const uint64_t bits_a, const uint64_t bits_b, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits_a > bits_b ? bits_a : bits_b);

	mpz_urandomb(a, state, bits_a);
	mpz_urandomb(b, state, bits_b);

	start_measurement(&measurement);
	for (size_t i = 0; i < SUB_ITERATIONS; i++)
	{
		mpz_sub(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	// Free memory
	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, SUB_ITERATIONS);
}

uint64_t GMPRandomMultiplication(testResult result, const uint64_t bits, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits * 2);

	mpz_urandomb(a, state, bits);
	mpz_urandomb(b, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < MUL_ITERATIONS; i++)
	{
		mpz_mul(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	// Free memory
	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, MUL_ITERATIONS);
}

uint64_t GMPBitShiftLeft(testResult result, const uint64_t bits, unsigned int seed, const unsigned long shift_bits)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a;
	mpz_t c;
	mpz_init(a);
	mpz_init2(c, bits + shift_bits + 64);

	mpz_urandomb(a, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < SHIFT_ITERATIONS; i++)
	{
		mpz_mul_2exp(c, a, shift_bits);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.gmp_result, 16, c);
	
	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, SHIFT_ITERATIONS);
}

uint64_t GMPBitShiftRight(testResult result, const uint64_t bits, unsigned int seed, const unsigned long shift_bits)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a;
	mpz_t c;
	mpz_init(a);
	mpz_init2(c, bits);

	mpz_urandomb(a, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < SHIFT_ITERATIONS; i++)
	{
		mpz_tdiv_q_2exp(c, a, shift_bits);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.gmp_result, 16, c);

	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, SHIFT_ITERATIONS);
}

uint64_t GMPAnd(testResult result, const uint64_t bits, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits + 64);

	mpz_urandomb(a, state, bits);
	mpz_urandomb(b, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_and(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}

uint64_t GMPOr(testResult result, const uint64_t bits, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits + 64);

	mpz_urandomb(a, state, bits);
	mpz_urandomb(b, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_ior(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}

uint64_t GMPXor(testResult result, const uint64_t bits, unsigned int seed)
{
	__measurement measurement;
	// Random number init
	time_t rawtime;
	time(&rawtime);
	gmp_randstate_t state;
	gmp_randinit_default(state);
	gmp_randseed_ui(state, rawtime + (seed << 2));

	mpz_t a, b;
	mpz_t c;
	mpz_init(a);
	mpz_init(b);
	mpz_init2(c, bits + 64);

	mpz_urandomb(a, state, bits);
	mpz_urandomb(b, state, bits);

	start_measurement(&measurement);
	for (size_t i = 0; i < LOGIC_ITERATIONS; i++)
	{
		mpz_xor(c, a, b);
	}
	stop_measurement(&measurement);

	mpz_get_str(result.a, 16, a);
	mpz_get_str(result.b, 16, b);
	mpz_get_str(result.gmp_result, 16, c);

	gmp_randclear(state);
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(c);

	return get_measurement_microseconds(&measurement, LOGIC_ITERATIONS);
}

void CompareNumbers(char * error, const char * gmp, const char * cuda)
{
	int precision = 20;
	mpf_t g, c;
	mpf_t res;
	mpf_init(g);
	mpf_init(c);
	mpf_init(res);

	mpf_set_str(g, gmp, 16);
	mpf_set_str(c, cuda, 16);

	mpf_div(res, g, c);

	gmp_sprintf(error, "%.*Ff", precision, res);

	mpf_clear(g);
	mpf_clear(c);
	mpf_clear(res);
}

void TestGMP(char * result_out, const char * hex_a, const char * hex_b)
{
	// Variables
	mpz_t a, b;
	mpz_t result;
	char * a_out = (char*)malloc(1024 * sizeof(char));
	char * b_out = (char*)malloc(1024 * sizeof(char));
	//char * result_out = (char*)malloc(1024 * sizeof(char));

	// Memory init
	mpz_init(a);
	mpz_init(b);
	mpz_init(result);

	// Set operands
	mpz_set_str(a, hex_a, 16);
	mpz_set_str(b, hex_b, 16);

	// Calculate and print result
	mpz_mul(result, a, b);

	printf("mpz_get_str(a): %s\n", mpz_get_str(a_out, 16, a));
	printf("mpz_get_str(b): %s\n", mpz_get_str(b_out, 16, b));
	printf("mpz_get_str(result): %s\n", mpz_get_str(result_out, 16, result));

	// Free memory
	mpz_clear(a);
	mpz_clear(b);
	mpz_clear(result);
	free(a_out);
	free(b_out);
	//free(result_out);

	//printf("END OF GMP TESTS\n\n");
}
