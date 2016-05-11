# cuGMP
CUDA library used to assess performance of CUDA GPGPU against popular multi precission arithmetic library GMP.

### TestApp

Concole application to make measurements comparing GMP and cuGMP performance.

### Getting started on Windows with Visual Studio

- Download and install CUDA toolkit from [nvidia.com](https://developer.nvidia.com/cuda-downloads).
- Download MPIR (GMP fork for Windows/VS) from [mpir.org](http://www.mpir.org/) (Tested with version 2.7.2)
- Extract MPIR to repository directory (or anywhere else, but then edit paths in TestApp project properties - Additional Library Directories and Additional Include Directories accordingly)
- In MPIR directory, open build.vc12\mpir.sln (for VS2013) and build lib\_mpir\_cxx and lib\_mpir\_gc in your desired configuration (or use batch build and build debug and release configurations)

Now you should be able to start using cuGMP.sln

### Getting started on Linux with Makefile

- Install cuda toolkit
- Install GMP library from your distribution repository or build from sources available at [gmplib.org](http://gmplib.org)
- Open makefile and check all include paths (primarily CCCUDAINCL variable)

**make** builds cuGMP library and TestApp

**make cuGMP.a** builds only cuGMP library
