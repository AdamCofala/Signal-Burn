CUDA_PATH ?= /usr/local/cuda-11.4
NVCC := $(CUDA_PATH)/bin/nvcc
CCBIN := /usr/bin/gcc-10

NVCCFLAGS := -O3 -arch=sm_35 -ccbin $(CCBIN) -allow-unsupported-compiler

LIBS := -lcufft -lstdc++

TARGET_DIR := bin
TARGET := $(TARGET_DIR)/dgp_core

all: $(TARGET)

$(TARGET): src/main.cu
	@mkdir -p $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) src/main.cu -o $(TARGET) $(LIBS)

clean:
	rm -rf $(TARGET_DIR)/*

.PHONY: all clean