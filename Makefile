.PHONY: all clean debug debug_asan release test test-stress test-complete test_persistance_100m_extension libwal_extension librecovery_extension test_integration_complete_extension test-extensions-with-logs prepare-logs-extensions validate-forensic-extensions-complete
all: $(TARGET)

CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE -D_POSIX_C_SOURCE=199309L -I./src/debug
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOG_DIR = logs

# Advanced calculations
ADVANCED_CALC_SOURCES = $(wildcard src/advanced_calculations/*.c)
MATH_RESEARCH_SOURCES = src/advanced_calculations/mathematical_research_engine.c

# Complex modules
COMPLEX_MODULES_SOURCES = src/complex_modules/realtime_analytics.c \
	                 src/complex_modules/distributed_computing.c \
	                 src/complex_modules/ai_optimization.c

# TSP, Knapsack, Collatz modules
TSP_MODULE_SOURCE = src/advanced_calculations/tsp_optimizer.c
KNAPSACK_MODULE_SOURCE = src/advanced_calculations/knapsack_optimizer.c
COLLATZ_MODULE_SOURCE = src/advanced_calculations/collatz_analyzer.c

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
HOMOMORPHIC_OBJ = $(OBJ_DIR)/crypto/homomorphic_encryption.o
DATA_PERSIST_OBJ = $(OBJ_DIR)/persistence/data_persistence.o
MEMORY_TRACKER_OBJ = $(OBJ_DIR)/debug/memory_tracker.o

# New objects for TSP, Knapsack, Collatz
TSP_OBJ = $(OBJ_DIR)/advanced_calculations/tsp_optimizer.o
KNAPSACK_OBJ = $(OBJ_DIR)/advanced_calculations/knapsack_optimizer.o
COLLATZ_OBJ = $(OBJ_DIR)/advanced_calculations/collatz_analyzer.o

# LUM Instant Displacement Module
LUM_INSTANT_DISPLACEMENT_OBJ = $(OBJ_DIR)/spatial/lum_instant_displacement.o

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
	  $(SRC_DIR)/crypto/homomorphic_encryption.c \
	  $(SRC_DIR)/persistence/data_persistence.c \
	  $(SRC_DIR)/debug/memory_tracker.c \
	  $(SRC_DIR)/debug/forensic_logger.c \
	  $(SRC_DIR)/file_formats/lum_native_file_handler.c \
	  $(SRC_DIR)/file_formats/lum_secure_serialization.c \
	  $(SRC_DIR)/complex_modules/ai_dynamic_config_manager.c \
	  $(SRC_DIR)/file_formats/lum_native_universal_format.c \
      $(ADVANCED_CALC_SOURCES) \
      $(MATH_RESEARCH_SOURCES) \
      $(COMPLEX_MODULES_SOURCES)

# Add LUM Instant Displacement Source
SOURCES += $(SRC_DIR)/spatial/lum_instant_displacement.c

OBJECTS = obj/main.o obj/lum/lum_core.o obj/vorax/vorax_operations.o obj/parser/vorax_parser.o \
	  obj/binary/binary_lum_converter.o obj/logger/lum_logger.o \
	  obj/optimization/memory_optimizer.o obj/optimization/pareto_optimizer.o \
	  obj/optimization/pareto_inverse_optimizer.o obj/optimization/simd_optimizer.o \
	  obj/optimization/zero_copy_allocator.o \
	  obj/parallel/parallel_processor.o obj/metrics/performance_metrics.o \
	  obj/crypto/crypto_validator.o obj/crypto/homomorphic_encryption.o obj/persistence/data_persistence.o \
	  obj/debug/memory_tracker.o \
	  obj/file_formats/lum_native_file_handler.o \
	  obj/file_formats/lum_secure_serialization.o \
	  obj/complex_modules/ai_dynamic_config_manager.o \
	  obj/file_formats/lum_native_universal_format.o \
	  obj/advanced_calculations/matrix_calculator.o \
	  obj/advanced_calculations/quantum_simulator.o \
	  obj/advanced_calculations/neural_network_processor.o \
	  obj/advanced_calculations/image_processor.o \
	  obj/advanced_calculations/audio_processor.o \
	  obj/advanced_calculations/mathematical_research_engine.o \
	  obj/complex_modules/realtime_analytics.o \
	  obj/complex_modules/distributed_computing.o \
	  obj/complex_modules/ai_optimization.o \
	  obj/debug/forensic_logger.o

# Add new objects to the list of all objects
OBJECTS += $(TSP_OBJ) $(KNAPSACK_OBJ) $(COLLATZ_OBJ) $(LUM_INSTANT_DISPLACEMENT_OBJ)

# Optimization objects
OPTIMIZATION_OBJS = $(OBJ_DIR)/optimization/memory_optimizer.o \
	            $(OBJ_DIR)/optimization/pareto_optimizer.o \
	            $(OBJ_DIR)/optimization/pareto_inverse_optimizer.o

EXECUTABLE = $(BIN_DIR)/lum_vorax
TARGET = $(EXECUTABLE)
LDFLAGS = -lpthread -lm -lrt -lm
SANITIZER_FLAGS = -fsanitize=address

# Create object directories
OBJ_DIRS = obj/lum obj/vorax obj/parser obj/binary obj/logger obj/optimization obj/parallel obj/metrics obj/crypto obj/persistence obj/debug obj/advanced_calculations obj/complex_modules obj/file_formats obj/spatial

$(OBJ_DIR):
	mkdir -p $(OBJ_DIRS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# Main executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -pthread -c $< -o $@

# Compilation with sanitizers
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

debug_asan: CFLAGS += -fsanitize=address,undefined -g -O1 -DDEBUG
debug_asan: LDFLAGS += -fsanitize=address,undefined
debug_asan: $(TARGET)

# Compilation optimisée pour production
release: CFLAGS += -O3 -DNDEBUG -march=native -flto
release: $(BIN_DIR)/lum_vorax

# Test specific zero-copy
test-zerocopy: $(BIN_DIR)/lum_vorax
	@echo "Test allocateur zero-copy..."
	@$(BIN_DIR)/lum_vorax --test-zerocopy-only

# Test million LUMs sécurisé
test-million-safe: debug
	@echo "Test million LUMs avec AddressSanitizer..."
	@./$(BIN_DIR)/lum_vorax --stress-million-safe

# Test de stress sécurisé pour Replit (exclure main.o pour éviter conflit)
STRESS_OBJECTS = $(filter-out $(OBJ_DIR)/main.o, $(OBJECTS))
$(BIN_DIR)/test_stress_safe: $(SRC_DIR)/tests/test_stress_safe.c $(STRESS_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(STRESS_OBJECTS) -lpthread -lm

# Test stress million LUMs selon prompt.txt
$(BIN_DIR)/test_million_lums: $(SRC_DIR)/tests/test_million_lums_stress.c $(STRESS_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(STRESS_OBJECTS) -lpthread -lm

# Test targets - COMPLETS TOUS MODULES
test: bin/test_lum_core bin/test_advanced_modules bin/test_unit_complete bin/test_integration_complete bin/test_regression_complete bin/test_advanced_complete
	@echo "Running all unit tests..."
	./bin/test_lum_core
	./bin/test_advanced_modules
	./bin/test_unit_complete
	./bin/test_integration_complete
	./bin/test_regression_complete
	./bin/test_advanced_complete

test-stress: bin/lum_vorax bin/test_stress_100m_all_modules
	@echo "Running comprehensive stress tests..."
	./bin/lum_vorax --stress-test-million
	./bin/test_stress_100m_all_modules

test-complete: test test-stress
	@echo "All tests completed successfully!"

# Tests unitaires complets
bin/test_unit_complete: $(OBJ_DIR)/tests/test_unit_lum_core_complete.o $(LUM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Tests intégration complets  
bin/test_integration_complete: $(OBJ_DIR)/tests/test_integration_complete.o $(LUM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Tests régression complets
bin/test_regression_complete: $(OBJ_DIR)/tests/test_regression_complete.o $(LUM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Tests avancés complets
bin/test_advanced_complete: $(OBJ_DIR)/tests/test_advanced_complete.o $(LUM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Tests stress 100M tous modules
bin/test_stress_100m_all_modules: $(OBJ_DIR)/tests/test_stress_100m_all_modules.o $(LUM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

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
$(OBJ_DIR)/debug/memory_tracker.o: src/debug/memory_tracker.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -c $< -o $@

# Link test executables
$(CORE_TEST_TARGET): $(SRC_DIR)/tests/test_core.c $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(ADVANCED_TEST_TARGET): $(SRC_DIR)/tests/test_advanced.c $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPLETE_TEST_TARGET): $(SRC_DIR)/tests/test_complete_functionality.c $(OBJECTS) $(MEMORY_TRACKER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Advanced calculations modules
obj/advanced_calculations/matrix_calculator.o: src/advanced_calculations/matrix_calculator.c src/advanced_calculations/matrix_calculator.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/matrix_calculator.c -o obj/advanced_calculations/matrix_calculator.o

obj/advanced_calculations/quantum_simulator.o: src/advanced_calculations/quantum_simulator.c src/advanced_calculations/quantum_simulator.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/quantum_simulator.c -o obj/advanced_calculations/quantum_simulator.o

obj/advanced_calculations/neural_network_processor.o: src/advanced_calculations/neural_network_processor.c src/advanced_calculations/neural_network_processor.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/neural_network_processor.c -o obj/advanced_calculations/neural_network_processor.o

# New TSP module compilation
$(TSP_OBJ): $(TSP_MODULE_SOURCE) src/advanced_calculations/matrix_calculator.h
	mkdir -p $(OBJ_DIR)/advanced_calculations
	$(CC) $(CFLAGS) -c $(TSP_MODULE_SOURCE) -o $(TSP_OBJ)

# New Knapsack module compilation
$(KNAPSACK_OBJ): $(KNAPSACK_MODULE_SOURCE) src/advanced_calculations/matrix_calculator.h
	mkdir -p $(OBJ_DIR)/advanced_calculations
	$(CC) $(CFLAGS) -c $(KNAPSACK_MODULE_SOURCE) -o $(KNAPSACK_OBJ)

# New Collatz module compilation
$(COLLATZ_OBJ): $(COLLATZ_MODULE_SOURCE) src/advanced_calculations/matrix_calculator.h
	mkdir -p $(OBJ_DIR)/advanced_calculations
	$(CC) $(CFLAGS) -c $(COLLATZ_MODULE_SOURCE) -o $(COLLATZ_OBJ)

# New multimedia and golden score modules compilation
obj/advanced_calculations/image_processor.o: src/advanced_calculations/image_processor.c src/advanced_calculations/image_processor.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/image_processor.c -o obj/advanced_calculations/image_processor.o

obj/advanced_calculations/audio_processor.o: src/advanced_calculations/audio_processor.c src/advanced_calculations/audio_processor.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/audio_processor.c -o obj/advanced_calculations/audio_processor.o

obj/advanced_calculations/video_processor.o: src/advanced_calculations/video_processor.c src/advanced_calculations/video_processor.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/video_processor.c -o obj/advanced_calculations/video_processor.o

obj/advanced_calculations/golden_score_optimizer.o: src/advanced_calculations/golden_score_optimizer.c src/advanced_calculations/golden_score_optimizer.h
	mkdir -p obj/advanced_calculations
	$(CC) $(CFLAGS) $(INCLUDES) -c src/advanced_calculations/golden_score_optimizer.c -o obj/advanced_calculations/golden_score_optimizer.o


# Complex modules
obj/complex_modules/realtime_analytics.o: src/complex_modules/realtime_analytics.c src/complex_modules/realtime_analytics.h
	mkdir -p obj/complex_modules
	$(CC) $(CFLAGS) $(INCLUDES) -c src/complex_modules/realtime_analytics.c -o obj/complex_modules/realtime_analytics.o

obj/complex_modules/distributed_computing.o: src/complex_modules/distributed_computing.c src/complex_modules/distributed_computing.h
	mkdir -p obj/complex_modules
	$(CC) $(CFLAGS) $(INCLUDES) -c src/complex_modules/distributed_computing.c -o obj/complex_modules/distributed_computing.o

obj/complex_modules/ai_optimization.o: src/complex_modules/ai_optimization.c src/complex_modules/ai_optimization.h
	mkdir -p obj/complex_modules
	$(CC) $(CFLAGS) $(INCLUDES) -c src/complex_modules/ai_optimization.c -o obj/complex_modules/ai_optimization.o

# New AI Dynamic Config Manager
obj/complex_modules/ai_dynamic_config_manager.o: src/complex_modules/ai_dynamic_config_manager.c src/complex_modules/ai_dynamic_config_manager.h
	mkdir -p obj/complex_modules
	$(CC) $(CFLAGS) $(INCLUDES) -c src/complex_modules/ai_dynamic_config_manager.c -o obj/complex_modules/ai_dynamic_config_manager.o

# New LUM Native Universal Format
obj/file_formats/lum_native_universal_format.o: src/file_formats/lum_native_universal_format.c src/file_formats/lum_native_universal_format.h
	mkdir -p obj/file_formats
	$(CC) $(CFLAGS) $(INCLUDES) -c src/file_formats/lum_native_universal_format.c -o obj/file_formats/lum_native_universal_format.o

# LUM Instant Displacement Module Compilation
$(LUM_INSTANT_DISPLACEMENT_OBJ): $(SRC_DIR)/spatial/lum_instant_displacement.c $(SRC_DIR)/spatial/lum_instant_displacement.h $(SRC_DIR)/lum/lum_core.h
	mkdir -p $(OBJ_DIR)/spatial
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean all test debug install uninstall

all: $(EXECUTABLE) | $(LOG_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR) *.o *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Legacy test target (replaced by comprehensive test target above)

forensic-validation:
	./ci/run_full_validation.sh

benchmark-baselines:
	./benchmark_baseline/run_all_benchmarks.sh

crypto-validation:
	./ci/run_crypto_validation.sh

memory-analysis:
	$(MAKE) debug_asan
	ASAN_OPTIONS=detect_leaks=1 ./bin/lum_vorax --stress-test-million &> asan_report.txt

# Advanced modules tests
test_advanced_modules: $(OBJ_DIR)/test_advanced_modules.o $(CORE_OBJS)
	$(CC) -o bin/$@ $^ $(LDFLAGS) -lpthread

# NOUVELLES cibles pour extensions (pas de duplication)
STRESS_100M_EXT_OBJS = $(OBJ_DIR)/test_stress_persistance_100m_extension.o
WAL_EXT_OBJS = $(OBJ_DIR)/transaction_wal_extension.o
RECOVERY_EXT_OBJS = $(OBJ_DIR)/recovery_manager_extension.o
PERSISTENCE_OBJS = $(OBJ_DIR)/data_persistence.o

# Tests stress persistance 100M+ (extension)
test_persistance_100m_extension: $(STRESS_100M_EXT_OBJS) $(CORE_OBJS) $(PERSISTENCE_OBJS)
	$(CC) -o bin/$@ $^ $(LDFLAGS) -lpthread

# Module WAL extension
libwal_extension: $(WAL_EXT_OBJS) $(PERSISTENCE_OBJS)
	@mkdir -p lib
	ar rcs lib/wal_extension.a $^

# Module recovery extension
librecovery_extension: $(RECOVERY_EXT_OBJS) $(WAL_EXT_OBJS) $(PERSISTENCE_OBJS)
	@mkdir -p lib
	ar rcs lib/recovery_extension.a $^

# Test intégration complète extension
test_integration_complete_extension: $(WAL_EXT_OBJS) $(RECOVERY_EXT_OBJS) $(STRESS_100M_EXT_OBJS) $(CORE_OBJS) $(PERSISTENCE_OBJS)
	$(CC) -o bin/$@ $^ $(LDFLAGS) -lpthread

# Compilation modules extensions
$(OBJ_DIR)/test_stress_persistance_100m_extension.o: src/tests/test_stress_persistance_100m_extension.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/transaction_wal_extension.o: src/persistence/transaction_wal_extension.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/recovery_manager_extension.o: src/persistence/recovery_manager_extension.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Tests extensions avec logs structurés
TEST_PERSISTANCE_EXT_COMPLETE = bin/test_persistance_100m_extension
TEST_WAL_EXT_COMPLETE = bin/test_wal_extension
TEST_RECOVERY_EXT_COMPLETE = bin/test_recovery_extension
TEST_INTEGRATION_EXT_COMPLETE = bin/test_integration_complete_extension

test-extensions-with-logs: prepare-logs-extensions $(TEST_PERSISTANCE_EXT_COMPLETE) $(TEST_WAL_EXT_COMPLETE) $(TEST_RECOVERY_EXT_COMPLETE) $(TEST_INTEGRATION_EXT_COMPLETE)
	@echo "🔬 Tests extensions avec logging forensique complet..."
	SESSION=$$(cat logs/current_session.txt); \
	./$(TEST_PERSISTANCE_EXT_COMPLETE) 2>&1 | tee logs/extensions/persistance/test_$${SESSION}.log; \
	./$(TEST_WAL_EXT_COMPLETE) 2>&1 | tee logs/extensions/wal/test_$${SESSION}.log; \
	./$(TEST_RECOVERY_EXT_COMPLETE) 2>&1 | tee logs/extensions/recovery/test_$${SESSION}.log; \
	./$(TEST_INTEGRATION_EXT_COMPLETE) 2>&1 | tee logs/extensions/integration/test_$${SESSION}.log

prepare-logs-extensions:
	@echo "🗂️ Préparation structure logs..."
	chmod +x prepare_logs_structure_extensions.sh
	./prepare_logs_structure_extensions.sh

# Validation forensique avec parsing automatique
validate-forensic-extensions-complete: test-extensions-with-logs
	@echo "🔍 Validation forensique avec parsing logs..."
	python3 tools/parse_extension_logs.py logs/extensions/ > logs/forensic/analysis_$$(cat logs/current_session.txt).json
	echo "📊 Résultats: logs/forensic/analysis_$$(cat logs/current_session.txt).json"

# Validation forensique complète (conservé)
validate-forensic-complete: test-extensions-complete
	@echo "🔬 Validation forensique extensions..."
	chmod +x validate_forensic_complete.sh
	./validate_forensic_complete.sh

# Variables pour compatibilité
SRCDIR = $(SRC_DIR)
BINDIR = $(BIN_DIR)

# Test targets
test-unit: $(BIN_DIR)/test_unit_lum_core_complete
	./$(BIN_DIR)/test_unit_lum_core_complete

# Legacy test-stress target (replaced by comprehensive test-stress target above)

test-displacement: $(BIN_DIR)/test_instant_displacement
	./$(BIN_DIR)/test_instant_displacement

$(BIN_DIR)/test_instant_displacement: $(SRC_DIR)/tests/test_instant_displacement.c $(SRC_DIR)/spatial/lum_instant_displacement.c $(SRC_DIR)/lum/lum_core.c $(SRC_DIR)/debug/memory_tracker.c $(SRC_DIR)/logger/lum_logger.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
