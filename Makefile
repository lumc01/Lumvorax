CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
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
EXECUTABLE = $(BINDIR)/lum_vorax

# Create object directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/lum $(OBJ_DIR)/vorax $(OBJ_DIR)/parser $(OBJ_DIR)/binary $(OBJ_DIR)/logger $(OBJ_DIR)/optimization $(OBJ_DIR)/parallel $(OBJ_DIR)/metrics $(OBJ_DIR)/crypto $(OBJ_DIR)/persistence

$(BINDIR):
	mkdir -p $(BINDIR)

# Build executable
$(EXECUTABLE): $(OBJECTS) | $(BINDIR)
	$(CC) $(OBJECTS) -o $@ -lpthread -lm

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile logger
$(OBJ_DIR)/logger/lum_logger.o: $(SRC_DIR)/logger/lum_logger.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile optimization module
$(OBJ_DIR)/optimization/memory_optimizer.o: $(SRC_DIR)/optimization/memory_optimizer.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile parallel processing module
$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c
	$(CC) $(CFLAGS) -c $< -o $@ -lpthread

# Compile metrics module
$(OBJ_DIR)/metrics/performance_metrics.o: $(SRC_DIR)/metrics/performance_metrics.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile crypto validation module
$(OBJ_DIR)/crypto/crypto_validator.o: $(SRC_DIR)/crypto/crypto_validator.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile persistence module
$(OBJ_DIR)/persistence/data_persistence.o: $(SRC_DIR)/persistence/data_persistence.c
	$(CC) $(CFLAGS) -c $< -o $@

# Targets
.PHONY: all clean run test install

all: $(EXECUTABLE)

clean:
	rm -rf $(OBJ_DIR) $(BINDIR) *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(EXECUTABLE)
	@echo "Running LUM/VORAX tests..."
	./$(EXECUTABLE)
	@echo "Tests completed!"

install: $(EXECUTABLE)
	cp $(EXECUTABLE) /usr/local/bin/

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: $(EXECUTABLE)

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
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build optimized release version"
	@echo "  install  - Install to /usr/local/bin"
	@echo "  help     - Show this help"