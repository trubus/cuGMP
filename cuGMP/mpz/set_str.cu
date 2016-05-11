#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuGMP.h"

void null_string(char * str, int length)
{
	for (size_t i = 0; i < length; i++)
	{
		str[i] = 0;
	}
}

void zero_string(char * str, int length)
{
	for (size_t i = 0; i < length; i++)
	{
		str[i] = '0';
	}

	// Null terminate string.
	str[length - 1] = 0;
}

int mpz_set_str_binary(mpz_ptr x, const char *str)
{
	char * c_limb = (char *) malloc(GMP_LIMB_BITS * sizeof(char) + 1);
	unsigned long length = (unsigned long)strlen(str);
	if (str[0] == '-') length--;
	unsigned long limbs;
	if (length % GMP_LIMB_BITS == 0)
	{
		limbs = length / GMP_LIMB_BITS;
	}
	else
	{
		limbs = length / GMP_LIMB_BITS + 1;
	}

	allocate_memory(x, limbs, limbs);
	
	const char * source = str;
	if (str[0] == '-')
	{
		source++;
		x->_mp_size = -1 * ABS(x->_mp_size);
	}

	// Read string from end to start (from least significant bits to most significant)
	const char * reader_end = source + length;
	for (unsigned long i = 0; i < limbs; i++)
	{
		reader_end -= GMP_LIMB_BITS;

		if (reader_end > source)
		{
			// Read 64 bits from input string.
			strncpy(c_limb, reader_end, GMP_LIMB_BITS);
			
			// Null terminate string.
			c_limb[GMP_LIMB_BITS] = 0;
		}
		else
		{
			// Read first (most significant) group of bits from string (smaller than 64 bit)
			int len = length - i * GMP_LIMB_BITS;
			strncpy(c_limb, source, len);
			
			// Null terminate string.
			c_limb[len] = 0;
		}

		// Convert limb from string of bits to unsigned long integer.
		x->_mp_d[i] = strtoull(c_limb, NULL, 2);
	}

	copy_operand_data_with_limbs(x, MemcpyDirection::memcpyHostToDevice);
	free(c_limb);
	return 0;
}

int mpz_set_str_hex(mpz_ptr x, const char *str)
{
	char * c_limb = (char *)malloc((GMP_LIMB_BITS / 4) * sizeof(char) + 1);
	unsigned long length = (unsigned long)strlen(str);
	if (str[0] == '-') length--;
	unsigned long limbs;
	if (length % GMP_LIMB_BITS == 0)
	{
		limbs = length / (GMP_LIMB_BITS / 4);
	}
	else
	{
		limbs = length / (GMP_LIMB_BITS / 4) + 1;
	}

	allocate_memory(x, limbs, limbs);

	const char * source = str;
	if (str[0] == '-')
	{
		source++;
		x->_mp_size = -1 * ABS(x->_mp_size);
	}

	// Read string from end to start (from least significant bits to most significant)
	const char * reader_end = source + length;
	for (unsigned long i = 0; i < limbs; i++)
	{
		reader_end -= GMP_LIMB_BITS / 4;

		if (reader_end > source)
		{
			// Read 64 bits from input string.
			strncpy(c_limb, reader_end, GMP_LIMB_BITS / 4);

			// Null terminate string.
			c_limb[GMP_LIMB_BITS / 4] = 0;
		}
		else
		{
			// Read first (most significant) group of bits from string (smaller than 64 bit)
			int len = length - i * (GMP_LIMB_BITS / 4);
			strncpy(c_limb, source, len);

			// Null terminate string.
			c_limb[len] = 0;
		}

		// Convert limb from string of bits to unsigned long integer.
		x->_mp_d[i] = strtoull(c_limb, NULL, 16);
	}

	copy_operand_data_with_limbs(x, MemcpyDirection::memcpyHostToDevice);
	free(c_limb);
	return 0;
}

// Set the value of x from str, a null-terminated C string in base base.
// Bases 2 and 16 supported only.
// This function returns 0 if the entire string is a valid number in base base. Otherwise it returns -1.
int mpz_set_str(mpz_ptr x, const char *str, int base)
{
	if (base == 2)
	{
		return mpz_set_str_binary(x, str);
	}

	if (base == 16)
	{
		return mpz_set_str_hex(x, str);
	}

	return -1;
}