#pragma once

#include <stdint.h>

#define ADD_ITERATIONS 1
#define SUB_ITERATIONS 1
#define MUL_ITERATIONS 1
#define SHIFT_ITERATIONS 1
#define LOGIC_ITERATIONS 1

enum Operation
{
	addition = 0,
	multiplication = 1,
	bit_shift_left = 2,
	subtraction = 3,
	bit_shift_right = 4,
	and_op = 5,
	or_op = 6,
	xor_op = 7,
	subtraction_first_larger = 8,
	subtraction_second_larger = 9
};

typedef struct
{
	unsigned int i;
	uint64_t operandSize;
	char * a;
	char * b;
	char * gmp_result;
	char * cuda_result;
	char * error;
	const char * name;
	uint64_t gmp_micro_s;
	uint64_t cuda_micro_s;
} testResult;

typedef struct
{
	unsigned int iterations;
	uint64_t bit_size;
	unsigned long shift_bits;
	Operation operation;
} testSetup;

bool CompareStrings(const char * gmp, const char * cuda);

void TestAddition(const int testSize, const uint64_t bits);
void TestSubtraction(const int testSize, const uint64_t bits);
void TestSubtractionFirstLarger(const int testSize, const uint64_t bits);
void TestSubtractionSecondLarger(const int testSize, const uint64_t bits);
void TestMultiplication(const int testSize, const uint64_t bits);
void TestBitShiftLeft(const int testSize, const uint64_t bits, const unsigned long shift_bits);
void TestBitShiftRight(const int testSize, const uint64_t bits, const unsigned long shift_bits);
void TestAnd(const int testSize, const uint64_t bits);
void TestOr(const int testSize, const uint64_t bits);
void TestXor(const int testSize, const uint64_t bits);
