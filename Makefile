CC = g++
CUDA_PATH ?= /usr/local/cuda-11.0
NVCC = $(CUDA_PATH)/bin/nvcc

CFLAGS = -mavx512f
LIB = libxdma.so
CPP_OBJS = XDMA.cpp.o XDMAController.cpp.o MemoryManager.cpp.o XDMACollective.cpp.o 
CU_OBJS = 
OBJS    = $(CPP_OBJS) $(CU_OBJS) 

LIBDIRS=-L/usr/local/cuda-11.0/lib64
INCDIRS=-I/usr/local/cuda-11.0/include

all:$(LIB)
	cp XDMA.h /usr/include
	cp XDMAController.h /usr/include
	cp MemoryManager.h /usr/include
	cp XDMACollective.h /usr/include
	cp $(LIB) /usr/lib64

%.cpp.o : %.cpp
	$(CC) $(CFLAGS) -fpic -c $< -o $@ $(LIBDIRS) $(INCDIRS) -lgdrapi


$(LIB) : $(OBJS)
	rm -f $@
	$(CC) -shared -o $@ $(OBJS) -lgdrapi -lcuda
	rm -f $(OBJS)