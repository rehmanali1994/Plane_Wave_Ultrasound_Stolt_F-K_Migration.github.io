CUDADIR = /usr/local/cuda
NVCC = $(CUDADIR)/bin/nvcc
MATLABDIR = /usr/local/MATLAB/R2015a
MEXSRCS = fkmigCUDA.cu
MEXOBJS = $(MEXSRCS:.cu=.o)
MEXFUNS = $(MEXOBJS:.o=.mexa64)
CUFLAGS = -arch=sm_50 -O3 --use_fast_math -Xcompiler -fPIC #-Xptxas -v
INCLUDE = -I. -I$(MATLABDIR)/extern/include/ -I$(CUDADIR)/samples/common/inc/
MEX = $(MATLABDIR)/bin/mex

all: $(MEXSRCS) $(MEXOBJS) $(MEXFUNS)

%.mexa64: %.o
	$(NVCC) $(CUFLAGS) -dlink $< -o dlink.o; \
	$(MEX) -largeArrayDims -cxx $< dlink.o -lcudart -lcufft -L$(CUDADIR)/lib64/; \
	rm dlink.o

%.o: %.cu
	$(NVCC) $(INCLUDE) $(CUFLAGS) -dc $< -o $@

clean:
	rm -f *.o
