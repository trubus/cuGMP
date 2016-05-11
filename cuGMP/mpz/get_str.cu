#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../cuGMP.h"

uint64_t allocate_str(char ** str, const int base, const mpz_ptr x)
{
	uint64_t size;
	if (base == 2)
	{
		size = ((uint64_t)ABS(x->_mp_size)) * GMP_LIMB_BITS;
	}
	else if (base == 16)
	{
		size = ((uint64_t)ABS(x->_mp_size)) * GMP_LIMB_BITS / 4;
	}
	else
	{
		printf("Unsupported base received in mpz_get_str - only 2 and 16 are supported.");
	}

	// Place for signum
	size++;

	if (*str == NULL)
	{
		*str = (char *)malloc((size + 1) * sizeof(char));
	}

	// Init string with zeroes.
	for (uint64_t i = 0; i < size; i++)
	{
		(*str)[i] = '0';
	}
	// Null terminate string.
	(*str)[size] = 0;

	return size;
}

void mpz_get_str_binary(char ** str, const int base, const mpz_ptr x)
{
	char * c_limb = (char *)malloc((GMP_LIMB_BITS + 1) * sizeof(char));
	c_limb[GMP_LIMB_BITS] = 0;
	uint64_t size = allocate_str(str, base, x);

	copy_operand_data_with_limbs(x, MemcpyDirection::memcpyDeviceToHost);

	char * writer_end = *str + size;
	for (int i = 0; i < ABS(x->_mp_size); i++)
	{
#ifdef _WIN32
		_ui64toa(x->_mp_d[i], c_limb, 2);
#else
		for (int j = 0; j <= GMP_LIMB_BITS; j++)
		{
			c_limb[i] = 0;
		}
		for (uint64_t j = (1ull << (GMP_LIMB_BITS - 1)); j > 0; j >>= 1)
		{
			strcat(c_limb, ((x->_mp_d[i] & j) == j) ? "1" : "0");
		}
#endif

		size_t c_limb_len = strnlen(c_limb, GMP_LIMB_BITS);
		strncpy(writer_end - c_limb_len, c_limb, c_limb_len);

		writer_end -= GMP_LIMB_BITS;
	}

	free(c_limb);
}

void mpz_get_str_hex(char ** str, const int base, const mpz_ptr x)
{
	char * c_limb = (char *)malloc((GMP_LIMB_BITS / 4 + 1) * sizeof(char));
	c_limb[GMP_LIMB_BITS / 4] = 0;
	uint64_t size = allocate_str(str, base, x);

	copy_operand_data_with_limbs(x, MemcpyDirection::memcpyDeviceToHost);

	char * writer_end = *str + size;
	for (int i = 0; i < ABS(x->_mp_size); i++)
	{
#ifdef _WIN32
		_ui64toa(x->_mp_d[i], c_limb, 16);
#else
		sprintf(c_limb, "%lx", x->_mp_d[i]);
#endif

		size_t c_limb_len = strnlen(c_limb, GMP_LIMB_BITS / 4);
		strncpy(writer_end - c_limb_len, c_limb, c_limb_len);

		writer_end -= GMP_LIMB_BITS / 4;
	}

	free(c_limb);
}

char * strip_leading_zeroes(char * str, const mpz_ptr x)
{
	int i = 0;
	if (x->_mp_size < 0)
	{
		// Reserve space for signum
		i = 1;
		str[0] = '-';
	}

	int num_of_zeroes = i;
	while (str[num_of_zeroes] == '0')
	{
		num_of_zeroes++;
	}

	if (strlen(str) == num_of_zeroes)
	{
		num_of_zeroes--;
	}

	num_of_zeroes -= i;

	if (num_of_zeroes > 0)
	{
		while (str[i] = str[i + num_of_zeroes])
		{
			i++;
		}
	}

	return str;
}

char * mpz_get_str(char * str, const int base, const mpz_ptr x)
{
	if (base == 2)
	{
		mpz_get_str_binary(&str, base, x);
	}
	else if (base == 16)
	{
		mpz_get_str_hex(&str, base, x);
	}
	else
	{
		return NULL;
	}

	return strip_leading_zeroes(str, x);
}