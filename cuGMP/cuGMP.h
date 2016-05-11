#pragma once

#include <stdint.h>

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

#ifndef ABS
#define ABS(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef GMP_config
#  ifdef _WIN32 
#    ifdef _WIN64 
#      define _LONG_LONG_LIMB  1 
#      define GMP_LIMB_BITS   64 
#    else 
#      define GMP_LIMB_BITS   32 
#    endif 
#    define __GMP_BITS_PER_MP_LIMB  GMP_LIMB_BITS 
#    define SIZEOF_MP_LIMB_T (GMP_LIMB_BITS >> 3) 
#    define GMP_NAIL_BITS                       0 
#    define BYTES_PER_MP_LIMB  SIZEOF_MP_LIMB_T
#  endif 
#  ifdef __linux__
#    define GMP_LIMB_BITS    64
#  endif
// Unused variable
//#  define GMP_NUMB_BITS     (GMP_LIMB_BITS - GMP_NAIL_BITS)
#endif

/**
* CUDA memory copy types
*/
enum MemcpyDirection
{
	memcpyHostToDevice = 1,      /**< Host   -> Device */
	memcpyDeviceToHost = 2,      /**< Device -> Host */
};

// GMP Data type definitions.
// If possible, only a copy of GMP.h for best compatibility of the libraries.
#ifndef GMP_data_types
typedef uint64_t		mp_limb_t;
//typedef long long int	mp_limb_signed_t;
typedef mp_limb_t		mp_bitcnt_t;
typedef int64_t			mp_size_t;

/* Struct similar to __mpz_struct but intended specifically to hold device (CUDA) data */
typedef struct
{
	int _mp_alloc;
	int _mp_size;
	mp_limb_t *_mp_d;
} __dev_mpz_struct;

typedef struct
{
	int _mp_alloc;		/* Number of *limbs* allocated and pointed
						to by the _mp_d field.  */
	int _mp_size;			/* abs(_mp_size) is the number of limbs the
							last field points to.  If _mp_size is
							negative this is a negative number.  */
	mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
	__dev_mpz_struct *_dev_mp_struct; /* Device allocated MPZ */
} __mpz_struct;

typedef __mpz_struct mpz_t[1];
typedef __mpz_struct *mpz_ptr;
typedef mp_limb_t *		mp_ptr;
#endif

#ifndef cuGMP_helpers
#define THREAD_ID blockIdx.x * blockDim.x + threadIdx.x
#endif

#ifndef cuGMP_settings
#define BLOCK_SIZE 512

// If defined, cudaDeviceSynchronize is called after each kernel execution.
//#define EXPLICIT_SYNCHRONIZATION

// If defined, all kernel calls are printed with dimensions
//#define KERNEL_PRINT
#endif

#ifndef cuGMP_functions

// Simple print of allocated number in cuda.
void printOperandCuda(mpz_ptr x);

// Print available CUDA devices
void printDevices(void);

// First call to cufftPlan takes a lot of time, every other call is quick - this heats up CUDA for accurate measurement
void cuFFT_init(void);
// Cuda set device etc.
void cudaInit(void);
// Reset cuda device - erase state
void cudaReset(void);

// Cuda allocation of GMP-Z number
void mpz_init(mpz_ptr);

// Cuda allocation of GMP-Z number with number of bits to allocate specified
void mpz_init2(mpz_ptr, mp_bitcnt_t);

// Allocates memory on CUDA for limbs.
void allocate_memory(mpz_ptr x, size_t limbs, size_t size);

// Cuda free of initialized numbers
void mpz_clear(mpz_ptr x);

// Memory operatins
// Copy operand data (eg. _mp_size) in specified direction
void copy_operand_data_without_limbs(mpz_ptr x, MemcpyDirection direction);
// Copy operand with limbs
void copy_operand_data_with_limbs(mpz_ptr x, MemcpyDirection direction);


// Set MPZ from string in base
int mpz_set_str(mpz_ptr x, const char *str, int base);
// Get string in base from MPZ
char * mpz_get_str(char * str, int base, mpz_ptr x);

// Math functions:
void mpz_add(mpz_ptr res, mpz_ptr a, mpz_ptr b);
void mpz_sub(mpz_ptr res, mpz_ptr a, mpz_ptr b);
void mpz_mul(mpz_ptr res, mpz_ptr a, mpz_ptr b);
void mpz_mul_2exp(mpz_ptr res, mpz_ptr a, unsigned long int exponent);
void mpz_tdiv_q_2exp(mpz_ptr res, mpz_ptr a, unsigned long int exponent);
void mpz_and(mpz_ptr res, mpz_ptr a, mpz_ptr b);
void mpz_ior(mpz_ptr res, mpz_ptr a, mpz_ptr b);
void mpz_xor(mpz_ptr res, mpz_ptr a, mpz_ptr b);


#endif