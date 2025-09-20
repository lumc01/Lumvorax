
# Makefile LUM/VORAX - Compilation complète 44 modules
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -O2 -fPIC
LDFLAGS = -lm -lpthread

# Répertoires
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOG_DIR = logs

# Sous-répertoires sources
LUM_DIR = $(SRC_DIR)/lum
VORAX_DIR = $(SRC_DIR)/vorax
BINARY_DIR = $(SRC_DIR)/binary
PARSER_DIR = $(SRC_DIR)/parser
LOGGER_DIR = $(SRC_DIR)/logger
DEBUG_DIR = $(SRC_DIR)/debug
CRYPTO_DIR = $(SRC_DIR)/crypto
PERSISTENCE_DIR = $(SRC_DIR)/persistence
OPTIMIZATION_DIR = $(SRC_DIR)/optimization
PARALLEL_DIR = $(SRC_DIR)/parallel
METRICS_DIR = $(SRC_DIR)/metrics
ADVANCED_DIR = $(SRC_DIR)/advanced_calculations
COMPLEX_DIR = $(SRC_DIR)/complex_modules
NETWORK_DIR = $(SRC_DIR)/network
SPATIAL_DIR = $(SRC_DIR)/spatial
FILE_FORMATS_DIR = $(SRC_DIR)/file_formats
TESTS_DIR = $(SRC_DIR)/tests

# Sous-répertoires objets
OBJ_LUM_DIR = $(OBJ_DIR)/lum
OBJ_VORAX_DIR = $(OBJ_DIR)/vorax
OBJ_BINARY_DIR = $(OBJ_DIR)/binary
OBJ_PARSER_DIR = $(OBJ_DIR)/parser
OBJ_LOGGER_DIR = $(OBJ_DIR)/logger
OBJ_DEBUG_DIR = $(OBJ_DIR)/debug
OBJ_CRYPTO_DIR = $(OBJ_DIR)/crypto
OBJ_PERSISTENCE_DIR = $(OBJ_DIR)/persistence
OBJ_OPTIMIZATION_DIR = $(OBJ_DIR)/optimization
OBJ_PARALLEL_DIR = $(OBJ_DIR)/parallel
OBJ_METRICS_DIR = $(OBJ_DIR)/metrics
OBJ_ADVANCED_DIR = $(OBJ_DIR)/advanced_calculations
OBJ_COMPLEX_DIR = $(OBJ_DIR)/complex_modules
OBJ_NETWORK_DIR = $(OBJ_DIR)/network
OBJ_SPATIAL_DIR = $(OBJ_DIR)/spatial
OBJ_FILE_FORMATS_DIR = $(OBJ_DIR)/file_formats

# Objets principaux
LUM_OBJECTS = $(OBJ_LUM_DIR)/lum_core.o
VORAX_OBJECTS = $(OBJ_VORAX_DIR)/vorax_operations.o
BINARY_OBJECTS = $(OBJ_BINARY_DIR)/binary_lum_converter.o
PARSER_OBJECTS = $(OBJ_PARSER_DIR)/vorax_parser.o
LOGGER_OBJECTS = $(OBJ_LOGGER_DIR)/lum_logger.o $(OBJ_LOGGER_DIR)/log_manager.o
DEBUG_OBJECTS = $(OBJ_DEBUG_DIR)/memory_tracker.o $(OBJ_DEBUG_DIR)/forensic_logger.o $(OBJ_DEBUG_DIR)/ultra_forensic_logger.o
CRYPTO_OBJECTS = $(OBJ_CRYPTO_DIR)/crypto_validator.o
PERSISTENCE_OBJECTS = $(OBJ_PERSISTENCE_DIR)/data_persistence.o $(OBJ_PERSISTENCE_DIR)/transaction_wal_extension.o $(OBJ_PERSISTENCE_DIR)/recovery_manager_extension.o
OPTIMIZATION_OBJECTS = $(OBJ_OPTIMIZATION_DIR)/memory_optimizer.o $(OBJ_OPTIMIZATION_DIR)/pareto_optimizer.o $(OBJ_OPTIMIZATION_DIR)/pareto_inverse_optimizer.o $(OBJ_OPTIMIZATION_DIR)/simd_optimizer.o $(OBJ_OPTIMIZATION_DIR)/zero_copy_allocator.o
PARALLEL_OBJECTS = $(OBJ_PARALLEL_DIR)/parallel_processor.o
METRICS_OBJECTS = $(OBJ_METRICS_DIR)/performance_metrics.o
NETWORK_OBJECTS = $(OBJ_NETWORK_DIR)/hostinger_client.o $(OBJ_NETWORK_DIR)/hostinger_resource_limiter.o
SPATIAL_OBJECTS = $(OBJ_SPATIAL_DIR)/lum_instant_displacement.o
FILE_FORMATS_OBJECTS = $(OBJ_FILE_FORMATS_DIR)/lum_native_file_handler.o $(OBJ_FILE_FORMATS_DIR)/lum_native_universal_format.o $(OBJ_FILE_FORMATS_DIR)/lum_secure_serialization.o

# Objets modules avancés
ADVANCED_OBJECTS = $(OBJ_ADVANCED_DIR)/matrix_calculator.o \
                  $(OBJ_ADVANCED_DIR)/quantum_simulator.o \
                  $(OBJ_ADVANCED_DIR)/neural_network_processor.o \
                  $(OBJ_ADVANCED_DIR)/neural_blackbox_computer.o \
                  $(OBJ_ADVANCED_DIR)/neural_advanced_optimizers.o \
                  $(OBJ_ADVANCED_DIR)/neural_ultra_precision_architecture.o \
                  $(OBJ_ADVANCED_DIR)/neural_blackbox_ultra_precision_tests.o \
                  $(OBJ_ADVANCED_DIR)/audio_processor.o \
                  $(OBJ_ADVANCED_DIR)/image_processor.o \
                  $(OBJ_ADVANCED_DIR)/collatz_analyzer.o \
                  $(OBJ_ADVANCED_DIR)/tsp_optimizer.o \
                  $(OBJ_ADVANCED_DIR)/knapsack_optimizer.o \
                  $(OBJ_ADVANCED_DIR)/mathematical_research_engine.o \
                  $(OBJ_ADVANCED_DIR)/golden_score_optimizer.o \
                  $(OBJ_ADVANCED_DIR)/blackbox_universal_module.o

# Objets modules complexes
COMPLEX_OBJECTS = $(OBJ_COMPLEX_DIR)/realtime_analytics.o \
                 $(OBJ_COMPLEX_DIR)/distributed_computing.o \
                 $(OBJ_COMPLEX_DIR)/ai_optimization.o \
                 $(OBJ_COMPLEX_DIR)/ai_dynamic_config_manager.o

# Tous les objets
ALL_OBJECTS = $(LUM_OBJECTS) $(VORAX_OBJECTS) $(BINARY_OBJECTS) $(PARSER_OBJECTS) \
              $(LOGGER_OBJECTS) $(DEBUG_OBJECTS) $(CRYPTO_OBJECTS) $(PERSISTENCE_OBJECTS) \
              $(OPTIMIZATION_OBJECTS) $(PARALLEL_OBJECTS) $(METRICS_OBJECTS) $(ADVANCED_OBJECTS) \
              $(COMPLEX_OBJECTS) $(NETWORK_OBJECTS) $(SPATIAL_OBJECTS) $(FILE_FORMATS_OBJECTS)

# Exécutables principaux
MAIN_EXECUTABLE = $(BIN_DIR)/lum_vorax_simple
TEST_EXECUTABLES = $(BIN_DIR)/test_advanced_complete \
                  $(BIN_DIR)/test_integration_complete \
                  $(BIN_DIR)/test_stress_100m_all_modules \
                  $(BIN_DIR)/test_lum_core \
                  $(BIN_DIR)/test_forensic_all_modules

.PHONY: all clean test test-stress test-integration test-forensic

all: directories $(MAIN_EXECUTABLE) $(TEST_EXECUTABLES)

directories:
	@mkdir -p $(BIN_DIR) $(LOG_DIR) $(OBJ_LUM_DIR) $(OBJ_VORAX_DIR) $(OBJ_BINARY_DIR) \
	         $(OBJ_PARSER_DIR) $(OBJ_LOGGER_DIR) $(OBJ_DEBUG_DIR) $(OBJ_CRYPTO_DIR) \
	         $(OBJ_PERSISTENCE_DIR) $(OBJ_OPTIMIZATION_DIR) $(OBJ_PARALLEL_DIR) \
	         $(OBJ_METRICS_DIR) $(OBJ_ADVANCED_DIR) $(OBJ_COMPLEX_DIR) $(OBJ_NETWORK_DIR) \
	         $(OBJ_SPATIAL_DIR) $(OBJ_FILE_FORMATS_DIR) logs/tests/modules logs/tests/stress \
	         logs/tests/integration logs/forensic/modules logs/forensic/tests

# Compilation objets par module
$(OBJ_LUM_DIR)/%.o: $(LUM_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_VORAX_DIR)/%.o: $(VORAX_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_BINARY_DIR)/%.o: $(BINARY_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_PARSER_DIR)/%.o: $(PARSER_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_LOGGER_DIR)/%.o: $(LOGGER_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DEBUG_DIR)/%.o: $(DEBUG_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_CRYPTO_DIR)/%.o: $(CRYPTO_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_PERSISTENCE_DIR)/%.o: $(PERSISTENCE_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_OPTIMIZATION_DIR)/%.o: $(OPTIMIZATION_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_PARALLEL_DIR)/%.o: $(PARALLEL_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_METRICS_DIR)/%.o: $(METRICS_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_ADVANCED_DIR)/%.o: $(ADVANCED_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_COMPLEX_DIR)/%.o: $(COMPLEX_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_NETWORK_DIR)/%.o: $(NETWORK_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_SPATIAL_DIR)/%.o: $(SPATIAL_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_FILE_FORMATS_DIR)/%.o: $(FILE_FORMATS_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Exécutable principal
$(MAIN_EXECUTABLE): $(ALL_OBJECTS) $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) $(SRC_DIR)/main.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

# Tests
$(BIN_DIR)/test_lum_core: $(ALL_OBJECTS) $(TESTS_DIR)/test_lum_core.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_lum_core.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

$(BIN_DIR)/test_advanced_complete: $(ALL_OBJECTS) $(TESTS_DIR)/test_advanced_complete.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_advanced_complete.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

$(BIN_DIR)/test_integration_complete: $(ALL_OBJECTS) $(TESTS_DIR)/test_integration_complete.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_integration_complete.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

$(BIN_DIR)/test_stress_100m_all_modules: $(ALL_OBJECTS) $(TESTS_DIR)/test_stress_100m_all_modules.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_stress_100m_all_modules.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

$(BIN_DIR)/test_forensic_all_modules: $(ALL_OBJECTS) $(TESTS_DIR)/test_forensic_all_modules.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_forensic_all_modules.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

# Tests unitaires complets 44 modules
$(BIN_DIR)/test_all_44_modules_complete: $(ALL_OBJECTS) $(TESTS_DIR)/test_all_44_modules_complete.c
	$(CC) $(CFLAGS) $(TESTS_DIR)/test_all_44_modules_complete.c $(ALL_OBJECTS) -o $@ $(LDFLAGS)

# Tests unitaires
test: $(TEST_EXECUTABLES) $(BIN_DIR)/test_all_44_modules_complete
	@echo "=== EXÉCUTION TESTS UNITAIRES 44 MODULES ==="
	@if [ -f $(BIN_DIR)/test_all_44_modules_complete ]; then \
		echo "🚀 Test complet des 44 modules:"; \
		$(BIN_DIR)/test_all_44_modules_complete | tee logs/tests/modules/all_44_modules.log; \
	fi
	@for test in $(TEST_EXECUTABLES); do \
		if [ -f $$test ]; then \
			echo "Exécution $$test"; \
			$$test | tee logs/tests/modules/$$(basename $$test).log; \
		fi; \
	done

# Tests stress
test-stress: $(BIN_DIR)/test_stress_100m_all_modules
	@echo "=== TESTS STRESS 100M ÉLÉMENTS ==="
	./$(BIN_DIR)/test_stress_100m_all_modules | tee logs/tests/stress/100m_all_modules.log

# Tests intégration
test-integration: $(BIN_DIR)/test_integration_complete
	@echo "=== TESTS INTÉGRATION COMPLÈTE ==="
	./$(BIN_DIR)/test_integration_complete | tee logs/tests/integration/complete.log

# Tests forensiques
test-forensic: $(BIN_DIR)/test_forensic_all_modules
	@echo "=== TESTS FORENSIQUES TOUS MODULES ==="
	./$(BIN_DIR)/test_forensic_all_modules | tee logs/forensic/tests/all_modules.log

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR)
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR)
