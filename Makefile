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

# Create object directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/lum $(OBJ_DIR)/vorax $(OBJ_DIR)/parser $(OBJ_DIR)/binary $(OBJ_DIR)/logger $(OBJ_DIR)/optimization $(OBJ_DIR)/parallel $(OBJ_DIR)/metrics $(OBJ_DIR)/crypto $(OBJ_DIR)/persistence

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build executable
$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ -lpthread -lm

# Generic rule for all object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rules for modules requiring special flags
$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ -pthread

$(OBJ_DIR)/crypto/crypto_validator.o: $(SRC_DIR)/crypto/crypto_validator.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ -lm

# Targets
.PHONY: all clean run test test-complete

all: $(EXECUTABLE)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(OBJECTS) # Assuming test executable needs all main objects
	@echo "Running LUM/VORAX tests..."
	# This is a placeholder, a proper test executable would be linked here.
	# For now, we assume main executable has test functionality or will be run.
	./$(EXECUTABLE)
	@echo "Tests completed!"

# Test complet de fonctionnalité
test-complete: $(OBJECTS) $(OBJ_DIR)/tests/test_complete_functionality.o
	$(CC) $(CFLAGS) -o bin/test_complete_functionality $(OBJECTS) $(OBJ_DIR)/tests/test_complete_functionality.o $(LDFLAGS)
	./bin/test_complete_functionality

$(OBJ_DIR)/tests/test_complete_functionality.o: src/tests/test_complete_functionality.c
	@mkdir -p $(OBJ_DIR)/tests
	$(CC) $(CFLAGS) -c $< -o $@

# Install
install: $(EXECUTABLE)
	cp $(EXECUTABLE) /usr/local/bin/

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: clean $(EXECUTABLE)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: clean $(EXECUTABLE)

# Show help
help:
	@echo "LUM/VORAX Build System"
	@echo "Available targets:"
	@echo "  all      - Build the LUM/VORAX system (default)"
	@echo "  clean    - Clean build files"
	@echo "  run      - Build and run the demo"
	@echo "  test     - Build and run tests"
	@echo "  test-complete - Build and run the complete functionality test"
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build optimized release version"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  help     - Show this help"