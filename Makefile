# Makefile for CUDA Batch Grayscale Converter
# Requires: CUDA Toolkit >= 11.0, NVIDIA NPP library

# ── Compiler & flags ────────────────────────────────────────────────────────
NVCC        := nvcc
CXX_FLAGS   := -std=c++17 -O2
NVCC_FLAGS  := $(CXX_FLAGS) -arch=sm_75   # Change sm_75 to match your GPU
INCLUDE     := -I./include
LIBS        := -lnppc -lnppi -lcudart

# ── Targets ─────────────────────────────────────────────────────────────────
TARGET      := grayscale_converter
SRC         := main.cu

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
	rm -rf output/

run: $(TARGET)
	./$(TARGET) --input data/images --output output --log output/run.log

help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run with default paths"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help"
