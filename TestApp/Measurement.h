#pragma once

#ifdef _WIN32
#include <windows.h>                // for Windows APIs
#else
#include <sys/time.h>
#endif

#include <stdint.h>

typedef struct
{
#ifdef _WIN32
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	LARGE_INTEGER frequency;
#else
	timeval start;
	timeval end;
#endif
} __measurement;

void start_measurement(__measurement * m);
void stop_measurement(__measurement * m);

uint64_t get_measurement_microseconds(__measurement * m);
uint64_t get_measurement_microseconds(__measurement * m, const unsigned int iterations);
uint64_t get_measurement_miliseconds(__measurement * m);
uint64_t get_measurement_miliseconds(__measurement * m, const unsigned int iterations);
uint64_t get_measurement_seconds(__measurement * m);
uint64_t get_measurement_seconds(__measurement * m, const unsigned int iterations);
