#pragma once

#include <stdint.h>

void PrintDevices(void);

uint64_t CUDAAddition(char * result, const char * hex_a, const char * hex_b);
uint64_t CUDASubtraction(char * result, const char * hex_a, const char * hex_b);
uint64_t CUDAMultiplication(char * result, const char * hex_a, const char * hex_b);
uint64_t CUDABitShiftLeft(char * result, const char * hex_a, unsigned long shift_bits);
uint64_t CUDABitShiftRight(char * result, const char * hex_a, unsigned long shift_bits);
uint64_t CUDAAnd(char * result, const char * hex_a, const char * hex_b);
uint64_t CUDAOr(char * result, const char * hex_a, const char * hex_b);
uint64_t CUDAXor(char * result, const char * hex_a, const char * hex_b);

void TestCuGMP(char * result, const char * hex_a, const char * hex_b);