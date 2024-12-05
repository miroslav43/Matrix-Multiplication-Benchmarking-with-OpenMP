# Makefile

CC = gcc-14
CFLAGS = -Wall -O2 -fopenmp -I$(SRC_DIR)
LDFLAGS = -fopenmp
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Ensure directories exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

# Source files
MAIN_SRC = $(SRC_DIR)/main.c
SERIAL_MULTIPLY_SRC = $(SRC_DIR)/serial_multiply.c
PARALLEL_MULTIPLY_SRC = $(SRC_DIR)/parallel_multiply.c
VALIDATION_SRC = $(SRC_DIR)/validation.c

# Header files
MATRIX_MULT_HDR = $(SRC_DIR)/matrix_mult.h
VALIDATION_HDR = $(SRC_DIR)/validation.h

# Object files
MAIN_OBJ = $(OBJ_DIR)/main.o
SERIAL_MULTIPLY_OBJ = $(OBJ_DIR)/serial_multiply.o
PARALLEL_MULTIPLY_OBJ = $(OBJ_DIR)/parallel_multiply.o
VALIDATION_OBJ = $(OBJ_DIR)/validation.o

# Program
PROGRAM = matrix_mult

all: $(BIN_DIR)/$(PROGRAM)

$(MAIN_OBJ): $(MAIN_SRC) $(MATRIX_MULT_HDR) $(VALIDATION_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SERIAL_MULTIPLY_OBJ): $(SERIAL_MULTIPLY_SRC) $(MATRIX_MULT_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(PARALLEL_MULTIPLY_OBJ): $(PARALLEL_MULTIPLY_SRC) $(MATRIX_MULT_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(VALIDATION_OBJ): $(VALIDATION_SRC) $(MATRIX_MULT_HDR) $(VALIDATION_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR)/$(PROGRAM): $(MAIN_OBJ) $(SERIAL_MULTIPLY_OBJ) $(PARALLEL_MULTIPLY_OBJ) $(VALIDATION_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(BIN_DIR)/$(PROGRAM)
