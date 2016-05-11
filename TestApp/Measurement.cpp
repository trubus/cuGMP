#include "Measurement.h"
#include <stdlib.h>

void start_measurement(__measurement * m)
{
#ifdef _WIN32
	QueryPerformanceFrequency(&m->frequency);
	QueryPerformanceCounter(&m->start);
#else
	gettimeofday(&m->start, NULL);
#endif
}

void stop_measurement(__measurement * m)
{
#ifdef _WIN32
	QueryPerformanceCounter(&m->end);
#else
	gettimeofday(&m->end, NULL);
#endif
}

uint64_t get_measurement_microseconds(__measurement * m)
{
#ifdef _WIN32
	return (m->end.QuadPart - m->start.QuadPart) * 1000000 / m->frequency.QuadPart;
#else
	return ((m->end.tv_sec * 1000000.0) + m->end.tv_usec) - ((m->start.tv_sec * 1000000.0) + m->start.tv_usec);
#endif
}

uint64_t get_measurement_microseconds(__measurement * m, const unsigned int iterations)
{
	return get_measurement_microseconds(m) / iterations;
}

uint64_t get_measurement_miliseconds(__measurement * m)
{
	return get_measurement_microseconds(m) / 1000;
}

uint64_t get_measurement_miliseconds(__measurement * m, const unsigned int iterations)
{
	return get_measurement_miliseconds(m) / iterations;
}

uint64_t get_measurement_seconds(__measurement * m)
{
	return get_measurement_microseconds(m) / 1000000;
}

uint64_t get_measurement_seconds(__measurement * m, const unsigned int iterations)
{
	return get_measurement_seconds(m) / iterations;
}
