#include "stdafx.h"

#include "Tests.h"
#include "TestGMP.h"
#include "TestCuGMP.h"

#ifndef _WIN32
#include <cstdlib>
#include <cstring>
#endif

#define BULK
//#define CMD
//#define DEVELOPMENT

#ifndef _WIN32
void strncpy_s(char * dest, size_t size, const char * src, size_t dsize)
{
		strncpy(dest, src, size);
}
#endif


void PrintUsageAndExit(void);
void PrintHeader(void);
void ExecuteOperation(testSetup setup);
testSetup ParseArguments(int argc, char *argv[]);

// Memory leak detector, using https://visualstudiogallery.msdn.microsoft.com/7c40a5d8-dd35-4019-a2af-cb1403f5939c
// BEWARE: cuFFT init takes about 10sec with vld.h active, without it it's about 0.5sec
//#include <vld.h>

// Possible input formats:
// operation block_size bit_length number_of_iterations (example: multiply 64 64 1000)

int main(int argc, char *argv[])
{
#ifdef BULK
	if (argc < 2)
	{
		PrintUsageAndExit();
	}

	PrintHeader();
	char *tokens[10];
	tokens[0] = 0;

	std::ifstream file(argv[1]);
	std::string line;
	while (std::getline(file, line))
	{
		std::string buffer;
		std::stringstream sline(line);
		int i = 1;
		while (sline >> buffer)
		{
			tokens[i] = (char *)calloc(256, sizeof(char));
			strncpy_s(tokens[i], 256, buffer.c_str(), buffer.size());
			i++;
		}
			
		testSetup setup = ParseArguments(i, tokens);
		ExecuteOperation(setup);

		for (int j = 0; j < i; j++)
		{
			free(tokens[j]);
		}
	}
#endif

#ifdef CMD
	testSetup setup = ParseArguments(argc, argv);
	
	PrintHeader();
	
	ExecuteOperation(setup);
#endif

#ifdef DEVELOPMENT
	/*char * hex_a = "5ebe38e6b5bf65ad2f48254e2a58f8fc00000000000000000000000000000000000";
	char * hex_b = "cd976019df1185c3e2669159311ea42e00000000000000000000000000000000000";*/
	char * hex_a = "3";
	char * hex_b = "FFFF";

	char * result_gmp = (char *)malloc(1048576 * sizeof(char));
	char * result_cuda = (char *)malloc(1048576 * sizeof(char));

	//TestGMP(result_gmp, hex_a, hex_b);
	//TestCuGMP(result_cuda, hex_a, hex_b);
	//CompareStrings(result_gmp, result_cuda);

	CUDASubtraction(result_cuda, hex_a, hex_b);

	printf("%s\n", result_cuda);

	free(result_cuda);
	free(result_gmp);
#endif
	//std::cin.get();
	return EXIT_SUCCESS;
}

void ExecuteOperation(testSetup setup)
{
	switch (setup.operation)
	{
	case Operation::addition:
		TestAddition(setup.iterations, setup.bit_size);
		break;
	case Operation::multiplication:
		TestMultiplication(setup.iterations, setup.bit_size);
		break;
	case Operation::bit_shift_left:
		TestBitShiftLeft(setup.iterations, setup.bit_size, setup.shift_bits);
		break;
	case Operation::subtraction:
		TestSubtraction(setup.iterations, setup.bit_size);
		break;
	case Operation::subtraction_first_larger:
		TestSubtractionFirstLarger(setup.iterations, setup.bit_size);
		break;
	case Operation::subtraction_second_larger:
		TestSubtractionSecondLarger(setup.iterations, setup.bit_size);
		break;
	case Operation::bit_shift_right:
		TestBitShiftRight(setup.iterations, setup.bit_size, setup.shift_bits);
		break;
	case Operation::and_op:
		TestAnd(setup.iterations, setup.bit_size);
		break;
	case Operation::or_op:
		TestOr(setup.iterations, setup.bit_size);
		break;
	case Operation::xor_op:
		TestXor(setup.iterations, setup.bit_size);
		break;
	default:
		printf("Operation not implemented\n");
		break;
	}
}

testSetup ParseArguments(int argc, char *argv[])
{
	if (argc < 4)
	{
		PrintUsageAndExit();
	}

	testSetup setup;
	char * end;

	// Parse operation
	if (strncmp(argv[1], "addition", 16) == 0)
	{
		setup.operation = Operation::addition;
	}
	else if (strncmp(argv[1], "subtraction", 16) == 0)
	{
		setup.operation = Operation::subtraction;
	}
	else if (strncmp(argv[1], "subtraction_first_larger", 32) == 0)
	{
		setup.operation = Operation::subtraction_first_larger;
	}
	else if (strncmp(argv[1], "subtraction_second_larger", 32) == 0)
	{
		setup.operation = Operation::subtraction_second_larger;
	}
	else if (strncmp(argv[1], "multiplication", 16) == 0)
	{
		setup.operation = Operation::multiplication;
	}
	else if (strncmp(argv[1], "bit_shift_left", 16) == 0)
	{
		setup.operation = Operation::bit_shift_left;
	}
	else if (strncmp(argv[1], "bit_shift_right", 16) == 0)
	{
		setup.operation = Operation::bit_shift_right;
	}
	else if (strncmp(argv[1], "and", 16) == 0)
	{
		setup.operation = Operation::and_op;
	}
	else if (strncmp(argv[1], "or", 16) == 0)
	{
		setup.operation = Operation::or_op;
	}
	else if (strncmp(argv[1], "xor", 16) == 0)
	{
		setup.operation = Operation::xor_op;
	}
	else
	{
		PrintUsageAndExit();
	}

	setup.iterations = strtoul(argv[2], &end, 10);
	if (argv[2] == end)
	{
		PrintUsageAndExit();
	}

	setup.bit_size = strtoull(argv[3], &end, 10);
	if (argv[3] == end)
	{
		PrintUsageAndExit();
	}

	if (argc == 5)
	{
		setup.shift_bits = strtoul(argv[4], &end, 10);
		if (argv[4] == end)
		{
			PrintUsageAndExit();
		}
	}
	else if (argc == 4 &&
		(setup.operation == Operation::bit_shift_left || setup.operation == Operation::bit_shift_right))
	{
		PrintUsageAndExit();
	}

	return setup;
}

void PrintUsageAndExit(void)
{
#ifdef BULK
	printf("Usage: TestApp.exe filepath\n");
	printf("File should contain commands in the following manner:\n");
	printf("operation iterations bit_size [shift_bits]\n");
	printf("Example: addition 1 128\n");
#else
	printf("Usage: TestApp.exe operation iterations bit_size [shift_bits]\n");
	printf("Example: TestApp.exe addition 1 128\n");
	printf("Example: TestApp.exe bit_shift_left 1 128 16\n");
#endif

	exit(EXIT_FAILURE);
}

void PrintHeader(void)
{
	PrintDevices();
	printf("Operation;Iteration;OperandSizeBits;RelativeError;GMPMicroSeconds;cuGMPMicroSeconds\n");
}
