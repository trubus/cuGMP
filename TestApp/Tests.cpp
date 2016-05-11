#include "stdafx.h"

#include "TestGMP.h"
#include "TestCuGMP.h"

#include "Tests.h"

#ifndef _WIN32
#include <cstdlib>
#include <cstring>
#endif

//#define DEBUG_PRINT
#define MIN(a,b) (a > b ? b : a)

#ifndef _WIN32
void strcpy_s(char * dest, size_t size, const char * src)
{
	strncpy(dest, src, size);
}
#endif

testResult NewTestResult(uint64_t bits, unsigned int i, const char * name)
{
	testResult result;
	result.cuda_micro_s = 0;
	result.gmp_micro_s = 0;
	result.operandSize = bits;
	result.i = i;
	result.name = name;
	return result;
}

void AllocateResultStrings(testResult *x, size_t stringSize, size_t resultStringSize)
{
	x->a = (char *)malloc(stringSize * sizeof(char));
	x->a[0] = 0;
	x->b = (char *)malloc(stringSize * sizeof(char));
	x->b[0] = 0;
	x->gmp_result = (char *)malloc(resultStringSize * sizeof(char));
	x->gmp_result[0] = 0;
	x->cuda_result = (char *)malloc(resultStringSize * sizeof(char));
	x->cuda_result[0] = 0;
	x->error = (char *)malloc(1024);
}

void FreeTestResult(testResult x)
{
	free(x.a);
	free(x.b);
	free(x.gmp_result);
	free(x.cuda_result);
	free(x.error);
}

bool CompareStrings(const char * gmp, const char * cuda)
{
	size_t size;
	size = strlen(gmp);
	unsigned int differences = 0;

	if (strlen(cuda) != size)
	{
		printf("Size difference\n");
		return false;
	}

	for (size_t i = 0; i < size; i++)
	{
		if (gmp[i] != cuda[i])
		{
			differences++;
		}
	}

	if (differences > 0)
	{
		printf("\tStrings differ at %d positions\n", differences);
		return false;
	}

	return true;
}

unsigned int StringDifference(const char * gmp, const char * cuda)
{
	size_t size;
	size = strlen(gmp);
	unsigned int difference = 0;

	difference += abs((long)(strlen(cuda) - size));

	for (size_t i = 0; i < MIN(strlen(gmp), strlen(cuda)); i++)
	{
		if (gmp[i] != cuda[i])
		{
			difference++;
		}
	}

	return difference;
}

bool ValidateResult(testResult x)
{
	if (!CompareStrings(x.gmp_result, x.cuda_result))
	{
		printf("       a: %s\n", x.a);
		printf("       b: %s\n", x.b);
		printf("     res: %s\n", x.gmp_result);
		printf("cuda_res: %s\n", x.cuda_result);
		return false;
	}

	return true;
}

void PrintResultTimes(testResult x, const char * operation)
{
	//printf("Iteration, OperandSizeBits, RelativeError, GMPMicroSeconds, cuGMPMicroSeconds\n");

	//printf("  GMP %s took %llu micro seconds\n", operation, x.gmp_micro_s);
	//printf("cuGMP %s took %llu micro seconds\n", operation, x.cuda_micro_s);
	
#ifdef _WIN32
	printf("%s;%d;%llu;%s;%llu;%llu\n", x.name, x.i, x.operandSize, x.error, x.gmp_micro_s, x.cuda_micro_s);
#else
	printf("%s;%d;%lu;%s;%lu;%lu\n", x.name, x.i, x.operandSize, x.error, x.gmp_micro_s, x.cuda_micro_s);
#endif
}


void TestAddition(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult addition = NewTestResult(bits, i, "addition");

		// Accomodate whole extra limb + terminate character.
		size_t stringSize = (bits + 64) / 4 + 1;
		// CUDA result has to be slightly bigger, because we can write several zeroes to it temporarily.
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&addition, stringSize, resultStringSize);

#ifdef DEBUG_PRINT
		printf("GMPRandomAddition ");
#endif

		addition.gmp_micro_s += GMPRandomAddition(addition, bits, i);

#ifdef DEBUG_PRINT
		printf("finished\nTestCUDAAddition");
#endif

		addition.cuda_micro_s += CUDAAddition(addition.cuda_result, addition.a, addition.b);

#ifdef DEBUG_PRINT
		printf("finished\n");
#endif

		if (ValidateResult(addition))
		{
			strcpy_s(addition.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(addition.error, addition.gmp_result, addition.cuda_result);
		}

		PrintResultTimes(addition, addition.name);
		FreeTestResult(addition);
	}
}

void TestSubtraction(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult subtraction = NewTestResult(bits, i, "subtraction");

		// Accomodate whole extra limb + terminate character.
		size_t stringSize = (bits + 64) / 4 + 1;
		// CUDA result has to be slightly bigger, because we can write several zeroes to it temporarily.
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&subtraction, stringSize, resultStringSize);

	#ifdef DEBUG_PRINT
		printf("GMPRandomSubtraction ");
	#endif

		subtraction.gmp_micro_s += GMPRandomSubtraction(subtraction, bits, bits, i);

	#ifdef DEBUG_PRINT
		printf("finished\nTestCUDASubtraction");
	#endif

		subtraction.cuda_micro_s += CUDASubtraction(subtraction.cuda_result, subtraction.a, subtraction.b);

	#ifdef DEBUG_PRINT
		printf("finished\n");
	#endif

		if (ValidateResult(subtraction))
		{
			strcpy_s(subtraction.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(subtraction.error, subtraction.gmp_result, subtraction.cuda_result);
		}

		PrintResultTimes(subtraction, subtraction.name);
		FreeTestResult(subtraction);
	}
}

void TestSubtractionFirstLarger(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult subtraction = NewTestResult(bits, i, "subtraction_first_larger");

		// Accomodate whole extra limb + terminate character.
		size_t stringSize = (bits + 64) / 4 + 1;
		// CUDA result has to be slightly bigger, because we can write several zeroes to it temporarily.
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&subtraction, stringSize, resultStringSize);

#ifdef DEBUG_PRINT
		printf("GMPRandomSubtraction ");
#endif
		// Second operand is 4 times smaller.
		subtraction.gmp_micro_s += GMPRandomSubtraction(subtraction, bits, bits / 16, i);

#ifdef DEBUG_PRINT
		printf("finished\nTestCUDASubtraction");
#endif

		subtraction.cuda_micro_s += CUDASubtraction(subtraction.cuda_result, subtraction.a, subtraction.b);

#ifdef DEBUG_PRINT
		printf("finished\n");
#endif

		if (ValidateResult(subtraction))
		{
			strcpy_s(subtraction.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(subtraction.error, subtraction.gmp_result, subtraction.cuda_result);
		}

		PrintResultTimes(subtraction, subtraction.name);
		FreeTestResult(subtraction);
	}
}

void TestSubtractionSecondLarger(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult subtraction = NewTestResult(bits, i, "subtraction_second_larger");

		// Accomodate whole extra limb + terminate character.
		size_t stringSize = (bits + 64) / 4 + 1;
		// CUDA result has to be slightly bigger, because we can write several zeroes to it temporarily.
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&subtraction, stringSize, resultStringSize);

#ifdef DEBUG_PRINT
		printf("GMPRandomSubtraction ");
#endif
		// Second operand is 4 times smaller.
		subtraction.gmp_micro_s += GMPRandomSubtraction(subtraction, bits / 16, bits, i);

#ifdef DEBUG_PRINT
		printf("finished\nTestCUDASubtraction");
#endif

		subtraction.cuda_micro_s += CUDASubtraction(subtraction.cuda_result, subtraction.a, subtraction.b);

#ifdef DEBUG_PRINT
		printf("finished\n");
#endif

		if (ValidateResult(subtraction))
		{
			strcpy_s(subtraction.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(subtraction.error, subtraction.gmp_result, subtraction.cuda_result);
		}

		PrintResultTimes(subtraction, subtraction.name);
		FreeTestResult(subtraction);
	}
}

void TestMultiplication(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult multiplication = NewTestResult(bits, i, "multiplication");
		size_t stringSize = (bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize * 2;

		AllocateResultStrings(&multiplication, stringSize, resultStringSize);

		multiplication.gmp_micro_s += GMPRandomMultiplication(multiplication, bits, i);
		multiplication.cuda_micro_s += CUDAMultiplication(multiplication.cuda_result, multiplication.a, multiplication.b);

	#ifdef DEBUG_PRINT
		printf("%s * %s =\n\t%s (GMP)\n\t%s (CUDA)\n", multiplication.a, multiplication.b, multiplication.gmp_result, multiplication.cuda_result);
	#endif

		CompareNumbers(multiplication.error, multiplication.gmp_result, multiplication.cuda_result);
		PrintResultTimes(multiplication, multiplication.name);
		FreeTestResult(multiplication);
	}
}

void TestBitShiftLeft(const int testSize, const uint64_t bits, const unsigned long shift_bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult result = NewTestResult(bits, i, "bit_shift_left");
		size_t stringSize = (bits + shift_bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&result, stringSize, resultStringSize);

		result.gmp_micro_s = GMPBitShiftLeft(result, bits, i, shift_bits);
		result.cuda_micro_s = CUDABitShiftLeft(result.cuda_result, result.a, shift_bits);

		if (ValidateResult(result))
		{
			strcpy_s(result.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(result.error, result.gmp_result, result.cuda_result);
		}

		PrintResultTimes(result, result.name);
		FreeTestResult(result);
	}
}

void TestBitShiftRight(const int testSize, const uint64_t bits, const unsigned long shift_bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult result = NewTestResult(bits, i, "bit_shift_right");
		size_t stringSize = (bits + shift_bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&result, stringSize, resultStringSize);

		result.gmp_micro_s = GMPBitShiftRight(result, bits, i, shift_bits);
		result.cuda_micro_s = CUDABitShiftRight(result.cuda_result, result.a, shift_bits);

		if (ValidateResult(result))
		{
			strcpy_s(result.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(result.error, result.gmp_result, result.cuda_result);
		}

		PrintResultTimes(result, result.name);
		FreeTestResult(result);
	}
}

void TestAnd(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult result = NewTestResult(bits, i, "and");
		size_t stringSize = (bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&result, stringSize, resultStringSize);

		result.gmp_micro_s = GMPAnd(result, bits, i);
		result.cuda_micro_s = CUDAAnd(result.cuda_result, result.a, result.b);

		if (ValidateResult(result))
		{
			strcpy_s(result.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(result.error, result.gmp_result, result.cuda_result);
		}

		PrintResultTimes(result, result.name);
		FreeTestResult(result);
	}
}

void TestOr(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult result = NewTestResult(bits, i, "or");
		size_t stringSize = (bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&result, stringSize, resultStringSize);

		result.gmp_micro_s = GMPOr(result, bits, i);
		result.cuda_micro_s = CUDAOr(result.cuda_result, result.a, result.b);

		if (ValidateResult(result))
		{
			strcpy_s(result.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(result.error, result.gmp_result, result.cuda_result);
		}

		PrintResultTimes(result, result.name);
		FreeTestResult(result);
	}
}

void TestXor(const int testSize, const uint64_t bits)
{
	for (int i = 0; i < testSize; i++)
	{
		testResult result = NewTestResult(bits, i, "xor");
		size_t stringSize = (bits + 64) / 4 + 1;
		size_t resultStringSize = stringSize + 64;

		AllocateResultStrings(&result, stringSize, resultStringSize);

		result.gmp_micro_s = GMPXor(result, bits, i);
		result.cuda_micro_s = CUDAXor(result.cuda_result, result.a, result.b);

		if (ValidateResult(result))
		{
			strcpy_s(result.error, 4, "1.0");
		}
		else
		{
			CompareNumbers(result.error, result.gmp_result, result.cuda_result);
		}

		PrintResultTimes(result, result.name);
		FreeTestResult(result);
	}
}
