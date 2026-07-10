CUDA_PATH ?= /usr/local/cuda-11.4
NVCC := $(CUDA_PATH)/bin/nvcc
CCBIN := /usr/bin/gcc-10

NVCCFLAGS := -O3 -arch=sm_35 -ccbin $(CCBIN) -allow-unsupported-compiler -Xcompiler -fPIC

LIBS := -lcufft -lstdc++

TARGET_DIR := bin
TARGET := $(TARGET_DIR)/libsb_core.so

all: $(TARGET)

$(TARGET): src/main.cu
	@mkdir -p $(TARGET_DIR)
	$(NVCC) $(NVCCFLAGS) -shared src/main.cu -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean