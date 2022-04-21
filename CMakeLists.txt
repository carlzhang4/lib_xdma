cmake_minimum_required(VERSION 3.7)

SET(CMAKE_C_COMPILER "/usr/bin/gcc-6")   # must be above project
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-6")


project(xdma LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)


set (CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC ")
set(CUDA_NVCC_FLAGS "")

add_definitions(-mavx512f)

set(SOURCE_FILES
	Communicator.cu
	XDMA.cpp
	XDMAController.cpp
	MemoryManager.cpp
	XDMACollective.cpp
)

set(HEAD_FILES 
	Communicator.cuh
	XDMA.h
	XDMAController.h
	MemoryManager.h
	XDMACollective.h
)


cuda_add_library(xdma SHARED ${SOURCE_FILES})

target_link_libraries(xdma
	${CUDA_LIBRARIES}
	pthread
	gdrapi
	cuda
    )

install(
	TARGETS xdma
	DESTINATION /usr/lib64
)

install(
	FILES ${HEAD_FILES}
	DESTINATION /usr/include)