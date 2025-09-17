CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE -D_POSIX_C_SOURCE=199309L -I./src/debug
LDFLAGS = -pthread -lm -lrt

# Core object files
CORE_OBJECTS = \
	obj/main.o \
	obj/lum/lum_core.o \
	obj/vorax/vorax_operations.o \
	obj/parser/vorax_parser.o \
	obj/binary/binary_lum_converter.o \
	obj/logger/lum_logger.o \
	obj/logger/log_manager.o \
	obj/optimization/memory_optimizer.o \
	obj/optimization/pareto_optimizer.o \
	obj/optimization/simd_optimizer.o \
	obj/parallel/parallel_processor.o \
	obj/metrics/performance_metrics.o \
	obj/crypto/crypto_validator.o \
	obj/persistence/data_persistence.o \
	obj/persistence/transaction_wal_extension.o \
	obj/persistence/recovery_manager_extension.o \
	obj/debug/memory_tracker.o \
	obj/debug/forensic_logger.o \
	obj/file_formats/lum_secure_serialization.o \
	obj/spatial/lum_instant_displacement.o

TARGET = bin/lum_vorax

all: $(TARGET)

# Create directories
obj bin:
	mkdir -p obj/lum obj/vorax obj/parser obj/binary obj/logger obj/optimization obj/parallel obj/metrics obj/crypto obj/persistence obj/debug obj/spatial obj/complex_modules obj/advanced_calculations obj/file_formats bin

$(TARGET): $(CORE_OBJECTS) | bin
	$(CC) $^ -o $@ $(LDFLAGS)

# Generic object file rule
obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

# Special rule for parallel processor (needs pthread)
obj/parallel/parallel_processor.o: src/parallel/parallel_processor.c | obj
	$(CC) $(CFLAGS) -pthread -c $< -o $@

# Rules for new modules
obj/file_formats/lum_secure_serialization.o: src/file_formats/lum_secure_serialization.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

# Rules for advanced calculations modules
obj/advanced_calculations/neural_blackbox_computer.o: src/advanced_calculations/neural_blackbox_computer.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/advanced_calculations/matrix_calculator.o: src/advanced_calculations/matrix_calculator.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

obj/spatial/lum_instant_displacement.o: src/spatial/lum_instant_displacement.c | obj
	$(CC) $(CFLAGS) -c $< -o $@


test_persistence_extensions: test_persistence_complete_extensions.c $(CORE_OBJECTS) | bin
	$(CC) $(CFLAGS) test_persistence_complete_extensions.c $(CORE_OBJECTS) -o bin/test_persistence_extensions $(LDFLAGS)

clean:
	rm -rf obj bin

.PHONY: all clean