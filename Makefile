CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE -D_POSIX_C_SOURCE=199309L -I./src/debug
LDFLAGS = -pthread -lm -lrt

# Core object files - COMPLET AVEC TOUS LES MODULES
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
        obj/optimization/pareto_inverse_optimizer.o \
        obj/optimization/zero_copy_allocator.o \
        obj/optimization/simd_optimizer.o \
        obj/parallel/parallel_processor.o \
        obj/metrics/performance_metrics.o \
        obj/crypto/crypto_validator.o \
        obj/crypto/homomorphic_encryption.o \
        obj/persistence/data_persistence.o \
        obj/persistence/transaction_wal_extension.o \
        obj/persistence/recovery_manager_extension.o \
        obj/debug/memory_tracker.o \
        obj/debug/forensic_logger.o \
        obj/file_formats/lum_secure_serialization.o \
        obj/file_formats/lum_native_file_handler.o \
        obj/file_formats/lum_native_universal_format.o \
        obj/spatial/lum_instant_displacement.o \
        obj/complex_modules/ai_dynamic_config_manager.o \
        obj/complex_modules/ai_optimization.o \
        obj/complex_modules/distributed_computing.o \
        obj/complex_modules/realtime_analytics.o \
        obj/network/hostinger_client.o \
        obj/network/hostinger_resource_limiter.o \
        obj/advanced_calculations/audio_processor.o \
        obj/advanced_calculations/blackbox_universal_module.o \
        obj/advanced_calculations/collatz_analyzer.o \
        obj/advanced_calculations/golden_score_optimizer.o \
        obj/advanced_calculations/image_processor.o \
        obj/advanced_calculations/knapsack_optimizer.o \
        obj/advanced_calculations/mathematical_research_engine.o \
        obj/advanced_calculations/matrix_calculator.o \
        obj/advanced_calculations/neural_advanced_optimizers.o \
        obj/advanced_calculations/neural_blackbox_computer.o \
        obj/advanced_calculations/neural_blackbox_ultra_precision_tests.o \
        obj/advanced_calculations/neural_network_processor.o \
        obj/advanced_calculations/neural_ultra_precision_architecture.o \
        obj/advanced_calculations/quantum_simulator.o \
        obj/advanced_calculations/tsp_optimizer.o

TARGET = bin/lum_vorax

all: $(TARGET)

# Create directories
obj bin:
        mkdir -p obj/lum obj/vorax obj/parser obj/binary obj/logger obj/optimization obj/parallel obj/metrics obj/crypto obj/persistence obj/debug obj/spatial obj/complex_modules obj/advanced_calculations obj/file_formats obj/network bin

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