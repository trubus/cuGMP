.SUFFIXES: .cpp .cu .o

CC = g++
CCFLAGS = -Wall -std=c++11
CCINCL = -I./cuGMP
CCCUDAINCL = -L/opt/cuda/lib64
CCLIBS = -lgmp -lcudart -lcufft
NVCC = nvcc
NVCCFLAGS = -std c++11 -gencode arch=compute_30,code=sm_30 --machine 64

CUGMP_LIB = cuGMP.a
EXECUTABLE = testApp

CUDA_SOURCES = $(wildcard cuGMP/*.cu cuGMP/mpz/*.cu cuGMP/kernels/*.cu)
CUDA_OBJECTS = $(patsubst %.cu,%.o, $(CUDA_SOURCES))
TESTAPP_SOURCES = $(wildcard TestApp/*.cpp)
TESTAPP_OBJECTS = $(patsubst %.cpp,%.o, $(TESTAPP_SOURCES))

all: $(EXECUTABLE)

$(EXECUTABLE): $(CUGMP_LIB) $(TESTAPP_OBJECTS) 
	${CC} $(TESTAPP_OBJECTS) $(CUGMP_LIB) $(CCCUDAINCL) $(CCLIBS) -o $(EXECUTABLE)

$(CUGMP_LIB): $(CUDA_OBJECTS)
	${NVCC} -lib $(CUDA_OBJECTS) -o $(CUGMP_LIB)

# Compiles all *.cu files, creating object files.
.cu.o:
	${NVCC} ${NVCCFLAGS} -c $< -o $@

# Compile all *.cpp files creating object files
.cpp.o:
	${CC} ${CCFLAGS} ${CCINCL} -c $< -o $@

clean: clean_testapp clean_cuda

clean_testapp:
	rm -f $(TESTAPP_OBJECTS) $(EXECUTABLE)

clean_cuda:
	rm -f $(CUDA_OBJECTS) $(CUGMP_LIB)
