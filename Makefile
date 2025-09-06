CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOG_DIR = logs

# Source files
SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/lum/lum_core.c \
          $(SRC_DIR)/vorax/vorax_operations.c \
          $(SRC_DIR)/parser/vorax_parser.c \
          $(SRC_DIR)/binary/binary_lum_converter.c \
          $(SRC_DIR)/logger/lum_logger.c \
          $(SRC_DIR)/optimization/memory_optimizer.c \
          $(SRC_DIR)/parallel/parallel_processor.c \
          $(SRC_DIR)/metrics/performance_metrics.c \
          $(SRC_DIR)/crypto/crypto_validator.c \
          $(SRC_DIR)/persistence/data_persistence.c

# Object files
OBJECTS = $(OBJ_DIR)/main.o \
          $(OBJ_DIR)/lum/lum_core.o \
          $(OBJ_DIR)/vorax/vorax_operations.o \
          $(OBJ_DIR)/parser/vorax_parser.o \
          $(OBJ_DIR)/binary/binary_lum_converter.o \
          $(OBJ_DIR)/logger/lum_logger.o \
          $(OBJ_DIR)/optimization/memory_optimizer.o \
          $(OBJ_DIR)/parallel/parallel_processor.o \
          $(OBJ_DIR)/metrics/performance_metrics.o \
          $(OBJ_DIR)/crypto/crypto_validator.o \
          $(OBJ_DIR)/persistence/data_persistence.o

# Executable
EXECUTABLE = $(BIN_DIR)/lum_vorax

# Create directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/lum $(OBJ_DIR)/vorax $(OBJ_DIR)/parser $(OBJ_DIR)/binary $(OBJ_DIR)/logger $(OBJ_DIR)/optimization $(OBJ_DIR)/parallel $(OBJ_DIR)/metrics $(OBJ_DIR)/crypto $(OBJ_DIR)/persistence

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# Build executable
$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ -lpthread -lm

# Generic compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Special compilation for pthread modules
$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -pthread -c $< -o $@

# Targets
.PHONY: all clean run test test-complete
.DEFAULT_GOAL := all

all: $(EXECUTABLE) | $(LOG_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR) *.o *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(EXECUTABLE)
	./$(EXECUTABLE) --test

test-complete: $(EXECUTABLE)
	./$(EXECUTABLE) --test-complete

# Verification targets
verify-build: $(EXECUTABLE)
	@echo "Build verification successful"
	@ls -la $(EXECUTABLE)

test-compilation: $(OBJ_DIR)
	@echo "Testing individual module compilation..."
	$(CC) $(CFLAGS) -c $(SRC_DIR)/lum/lum_core.c -o $(OBJ_DIR)/test_lum_core.o
	@echo "LUM Core: OK"
	$(CC) $(CFLAGS) -c $(SRC_DIR)/vorax/vorax_operations.c -o $(OBJ_DIR)/test_vorax_ops.o
	@echo "VORAX Operations: OK"
	rm -f $(OBJ_DIR)/test_*.o