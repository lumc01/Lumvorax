CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOG_DIR = logs

# Object files
MAIN_OBJ = $(OBJ_DIR)/main.o
LUM_CORE_OBJ = $(OBJ_DIR)/lum/lum_core.o
VORAX_OPS_OBJ = $(OBJ_DIR)/vorax/vorax_operations.o
PARSER_OBJ = $(OBJ_DIR)/parser/vorax_parser.o
BINARY_CONV_OBJ = $(OBJ_DIR)/binary/binary_lum_converter.o
LOGGER_OBJ = $(OBJ_DIR)/logger/lum_logger.o
MEMORY_OPT_OBJ = $(OBJ_DIR)/optimization/memory_optimizer.o
PARETO_OPT_OBJ = $(OBJ_DIR)/optimization/pareto_optimizer.o
PARETO_INV_OPT_OBJ = $(OBJ_DIR)/optimization/pareto_inverse_optimizer.o
SIMD_OPT_OBJ = $(OBJ_DIR)/optimization/simd_optimizer.o
ZERO_COPY_OBJ = $(OBJ_DIR)/optimization/zero_copy_allocator.o
PARALLEL_PROC_OBJ = $(OBJ_DIR)/parallel/parallel_processor.o
PERF_METRICS_OBJ = $(OBJ_DIR)/metrics/performance_metrics.o
CRYPTO_VAL_OBJ = $(OBJ_DIR)/crypto/crypto_validator.o
DATA_PERSIST_OBJ = $(OBJ_DIR)/persistence/data_persistence.o
MEMORY_TRACKER_OBJ = $(OBJ_DIR)/debug/memory_tracker.o

SOURCES = $(SRC_DIR)/main.c \
	  $(SRC_DIR)/lum/lum_core.c \
	  $(SRC_DIR)/vorax/vorax_operations.c \
	  $(SRC_DIR)/parser/vorax_parser.c \
	  $(SRC_DIR)/binary/binary_lum_converter.c \
	  $(SRC_DIR)/logger/lum_logger.c \
	  $(SRC_DIR)/optimization/memory_optimizer.c \
	  $(SRC_DIR)/optimization/pareto_optimizer.c \
	  $(SRC_DIR)/optimization/pareto_inverse_optimizer.c \
	  $(SRC_DIR)/optimization/simd_optimizer.c \
	  $(SRC_DIR)/optimization/zero_copy_allocator.c \
	  $(SRC_DIR)/parallel/parallel_processor.c \
	  $(SRC_DIR)/metrics/performance_metrics.c \
	  $(SRC_DIR)/crypto/crypto_validator.c \
	  $(SRC_DIR)/persistence/data_persistence.c \
	  $(SRC_DIR)/debug/memory_tracker.c

OBJECTS = $(OBJ_DIR)/main.o \
	  $(OBJ_DIR)/lum/lum_core.o \
	  $(OBJ_DIR)/vorax/vorax_operations.o \
	  $(OBJ_DIR)/parser/vorax_parser.o \
	  $(OBJ_DIR)/binary/binary_lum_converter.o \
	  $(OBJ_DIR)/logger/lum_logger.o \
	  $(OBJ_DIR)/optimization/memory_optimizer.o \
	  $(OBJ_DIR)/optimization/pareto_optimizer.o \
	  $(OBJ_DIR)/optimization/pareto_inverse_optimizer.o \
	  $(OBJ_DIR)/optimization/simd_optimizer.o \
	  $(OBJ_DIR)/optimization/zero_copy_allocator.o \
	  $(OBJ_DIR)/parallel/parallel_processor.o \
	  $(OBJ_DIR)/metrics/performance_metrics.o \
	  $(OBJ_DIR)/crypto/crypto_validator.o \
	  $(OBJ_DIR)/persistence/data_persistence.o \
	  $(OBJ_DIR)/debug/memory_tracker.o

# Optimization objects
OPTIMIZATION_OBJS = $(OBJ_DIR)/optimization/memory_optimizer.o \
	            $(OBJ_DIR)/optimization/pareto_optimizer.o \
	            $(OBJ_DIR)/optimization/pareto_inverse_optimizer.o

EXECUTABLE = $(BIN_DIR)/lum_vorax
TARGET = $(EXECUTABLE)
LDFLAGS = -lpthread -lm

# Create object directories
OBJ_DIRS = obj/lum obj/vorax obj/parser obj/binary obj/logger obj/optimization obj/parallel obj/metrics obj/crypto obj/persistence obj/debug

$(OBJ_DIR):
	mkdir -p $(OBJ_DIRS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# Main executable
$(TARGET): $(MAIN_OBJ) $(LUM_CORE_OBJ) $(VORAX_OPS_OBJ) $(PARSER_OBJ) $(BINARY_CONV_OBJ) $(LOGGER_OBJ) $(MEMORY_OPT_OBJ) $(PARETO_OPT_OBJ) $(PARETO_INV_OPT_OBJ) $(SIMD_OPT_OBJ) $(ZERO_COPY_OBJ) $(PARALLEL_PROC_OBJ) $(PERF_METRICS_OBJ) $(CRYPTO_VAL_OBJ) $(DATA_PERSIST_OBJ) $(MEMORY_TRACKER_OBJ) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -pthread -c $< -o $@

# Test de stress sécurisé pour Replit (exclure main.o pour éviter conflit)
STRESS_OBJECTS = $(filter-out $(OBJ_DIR)/main.o, $(OBJECTS))
$(BIN_DIR)/test_stress_safe: $(SRC_DIR)/tests/test_stress_safe.c $(STRESS_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(STRESS_OBJECTS) -lpthread -lm

# Test stress million LUMs selon prompt.txt
$(BIN_DIR)/test_million_lums: $(SRC_DIR)/tests/test_million_lums_stress.c $(STRESS_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(STRESS_OBJECTS) -lpthread -lm

# Test targets
test_core: $(CORE_TEST_TARGET)
	./$(CORE_TEST_TARGET)

test_advanced: $(ADVANCED_TEST_TARGET)
	./$(ADVANCED_TEST_TARGET)

# Complete functionality test (unified target)
test_complete: $(COMPLETE_TEST_TARGET)
	./$(COMPLETE_TEST_TARGET)

$(COMPLETE_TEST_TARGET): $(OBJ_DIR)/tests/test_complete_functionality.o $(LUM_CORE_OBJ) $(VORAX_OPS_OBJ) $(PARSER_OBJ) $(BINARY_CONV_OBJ) $(LOGGER_OBJ) $(MEMORY_OPT_OBJ) $(PARETO_OPT_OBJ) $(PARALLEL_PROC_OBJ) $(PERF_METRICS_OBJ) $(CRYPTO_VAL_OBJ) $(DATA_PERSIST_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

# Compilation of objects for tests
$(OBJ_DIR)/tests/test_complete_functionality.o: $(SRC_DIR)/tests/test_complete_functionality.c $(OBJECTS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation of objects for tests
$(OBJ_DIR)/tests/test_stress_safe.o: $(SRC_DIR)/tests/test_stress_safe.c $(STRESS_OBJECTS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation of objects for tests
$(OBJ_DIR)/tests/test_million_lums_stress.o: $(SRC_DIR)/tests/test_million_lums_stress.c $(STRESS_OBJECTS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation of debug objects
$(OBJ_DIR)/debug/memory_tracker.o: $(SRC_DIR)/debug/memory_tracker.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -c $< -o $@

# Link test executables
$(CORE_TEST_TARGET): $(SRC_DIR)/tests/test_core.c $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ADVANCED_TEST_TARGET): $(SRC_DIR)/tests/test_advanced.c $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPLETE_TEST_TARGET): $(SRC_DIR)/tests/test_complete_functionality.c $(OBJECTS) $(MEMORY_TRACKER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: all clean run test

all: $(EXECUTABLE) | $(LOG_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR) *.o *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(EXECUTABLE)
	./$(EXECUTABLE) --test