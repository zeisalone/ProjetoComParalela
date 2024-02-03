################################################################################
# Makefile for general code snippets
#
# by André Pereira
################################################################################

SHELL = /bin/sh

BIN_NAME_CUDA = MDcuda.exe
BIN_NAME_SEQ  = MDseq.exe


CXX = nvcc
LD  = nvcc
CC1 = gcc


CXXFLAGS   = -O2 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets
CFLAGS = -O3 -Ofast -funroll-loops -mavx

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

SRCCU = $(wildcard $(SRC_DIR)/*.cu)
SRC = src/
OBJ = $(patsubst src/%.cu,build/%.o,$(SRCCU))
BIN = $(BIN_NAME_CUDA)

vpath %.cu $(SRC_DIR)

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.o: %.cu
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BIN_DIR)/$(BIN_NAME_CUDA): $(OBJ)
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ) $(LIBS)

$(BIN_DIR)/$(BIN_NAME_SEQ): $(SRC)MDseq.cpp
	module load gcc/11.2.0;\
	$(CC1) $(CFLAGS) $(SRC)MDseq.cpp -lm -o $(BIN_DIR)/$(BIN_NAME_SEQ)


checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME_CUDA) $(BIN_DIR)/$(BIN_NAME_SEQ)
clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/*

run: $(BIN_DIR)/$(BIN_NAME_CUDA)
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	sbatch runcuda.sh


runcuda: $(BIN_DIR)/$(BIN_NAME_CUDA)
	module load gcc/7.2.0;\
	module load cuda/11.3.1;\
	nvprof $(BIN_DIR)/$(BIN_NAME_CUDA) < inputdata.txt