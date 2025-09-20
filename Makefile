
CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -ffast-math -funroll-loops -g -D_GNU_SOURCE -D_POSIX_C_SOURCE=199309L -I./src/debug -I./src -mavx512f -mfma
LDFLAGS = -pthread -lm -lrt

# TOUS les modules du système LUM/VORAX
COMPLETE_OBJECTS = \
	obj/main.o \
	obj/lum/lum_core.o \
	obj/vorax/vorax_operations.o \
	obj/parser/vorax_parser.o \
	obj/binary/binary_lum_converter.o \
	obj/logger/lum_logger.o \
	obj/logger/log_manager.o \
	obj/debug/memory_tracker.o \
	obj/debug/forensic_logger.o \
	obj/persistence/data_persistence.o \
	obj/persistence/transaction_wal_extension.o \
	obj/persistence/recovery_manager_extension.o \
	obj/crypto/crypto_validator.o \
	obj/optimization/memory_optimizer.o \
	obj/optimization/pareto_optimizer.o \
	obj/optimization/pareto_inverse_optimizer.o \
	obj/optimization/simd_optimizer.o \
	obj/optimization/zero_copy_allocator.o \
	obj/parallel/parallel_processor.o \
	obj/metrics/performance_metrics.o \
	obj/advanced_calculations/matrix_calculator.o \
	obj/advanced_calculations/quantum_simulator.o \
	obj/advanced_calculations/neural_network_processor.o \
	obj/advanced_calculations/audio_processor.o \
	obj/advanced_calculations/image_processor.o \
	obj/advanced_calculations/collatz_analyzer.o \
	obj/advanced_calculations/tsp_optimizer.o \
	obj/advanced_calculations/knapsack_optimizer.o \
	obj/advanced_calculations/mathematical_research_engine.o \
	obj/advanced_calculations/blackbox_universal_module.o \
	obj/advanced_calculations/neural_blackbox_computer.o \
	obj/advanced_calculations/golden_score_optimizer.o \
	obj/advanced_calculations/neural_advanced_optimizers.o \
	obj/advanced_calculations/neural_ultra_precision_architecture.o \
	obj/advanced_calculations/neural_blackbox_ultra_precision_tests.o \
	obj/complex_modules/realtime_analytics.o \
	obj/complex_modules/distributed_computing.o \
	obj/complex_modules/ai_optimization.o \
	obj/complex_modules/ai_dynamic_config_manager.o \
	obj/file_formats/lum_secure_serialization.o \
	obj/file_formats/lum_native_file_handler.o \
	obj/file_formats/lum_native_universal_format.o \
	obj/spatial/lum_instant_displacement.o \
	obj/network/hostinger_client.o \
	obj/network/hostinger_resource_limiter.o

TARGET = bin/lum_vorax_complete

all: $(TARGET)

obj bin logs:
	mkdir -p obj/{lum,vorax,parser,binary,logger,debug,persistence,crypto,optimization,parallel,metrics,advanced_calculations,complex_modules,file_formats,spatial,network} bin logs/{compilation,execution,tests,stress_tests,modules_tests,performance,diagnostics}

$(TARGET): $(COMPLETE_OBJECTS) | bin logs
	$(CC) $^ -o $@ $(LDFLAGS)

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj bin

# Tests complets tous modules
test-all: $(TARGET)
	echo "=== TESTS COMPLETS TOUS MODULES LUM/VORAX ==="
	./$(TARGET) --test-all-modules 2>&1 | tee logs/execution/all_modules_test_$(shell date +%Y%m%d_%H%M%S).log

test-stress: $(TARGET)
	echo "=== TEST STRESS 1M+ LUMs ==="
	./$(TARGET) --stress-test-million 2>&1 | tee logs/stress_tests/stress_1m_$(shell date +%Y%m%d_%H%M%S).log

test-advanced: $(TARGET)
	echo "=== TESTS MODULES AVANCÉS ==="
	./$(TARGET) --test-advanced 2>&1 | tee logs/modules_tests/advanced_test_$(shell date +%Y%m%d_%H%M%S).log

.PHONY: all clean test-all test-stress test-advanced
