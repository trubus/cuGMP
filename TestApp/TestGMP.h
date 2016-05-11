#pragma once

#include "Tests.h"
#include <stdint.h>

void CompareNumbers(char * error, const char * gmp, const char * cuda);

uint64_t GMPRandomAddition(testResult result, const uint64_t bits, unsigned int seed);
uint64_t GMPRandomSubtraction(testResult result, const uint64_t bits_a, const uint64_t bits_b, unsigned int seed);
uint64_t GMPRandomMultiplication(testResult result, const uint64_t bits, unsigned int seed);
uint64_t GMPBitShiftLeft(testResult result, const uint64_t bits, unsigned int seed, const unsigned long shift_bits);
uint64_t GMPBitShiftRight(testResult result, const uint64_t bits, unsigned int seed, const unsigned long shift_bits);
uint64_t GMPAnd(testResult result, const uint64_t bits, unsigned int seed);
uint64_t GMPOr(testResult result, const uint64_t bits, unsigned int seed);
uint64_t GMPXor(testResult result, const uint64_t bits, unsigned int seed);

void TestGMP(char * result, const char * hex_a, const char * hex_b);