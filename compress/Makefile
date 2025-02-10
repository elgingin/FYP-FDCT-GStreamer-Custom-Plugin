# CUDA settings
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXXFLAGS = -Xcompiler -fPIC -O3
INCLUDE = -I$(CUDA_PATH)/include -I../include
LIBS = -L$(CUDA_PATH)/lib64 -lcufft -lcudart

# Files
SRC = compress.cu
OBJ = compress.o

# Default target
all: $(OBJ)

# Compile compress.cu to object file
$(OBJ): $(SRC)
	$(NVCC) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJ)
