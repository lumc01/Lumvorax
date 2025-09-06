CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LOG_DIR = logs

SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/lum/lum_core.c \
          $(SRC_DIR)/vorax/vorax_operations.c \
          $(SRC_DIR)/parser/vorax_parser.c \
          $(SRC_DIR)/binary/binary_lum_converter.c \
          $(SRC_DIR)/logger/lum_logger.c \
          $(SRC_DIR)/optimization/memory_optimizer.c \
          $(SRC_DIR)/optimization/pareto_optimizer.c \
          $(SRC_DIR)/optimization/pareto_inverse_optimizer.c \
          $(SRC_DIR)/parallel/parallel_processor.c \
          $(SRC_DIR)/metrics/performance_metrics.c \
          $(SRC_DIR)/crypto/crypto_validator.c \
          $(SRC_DIR)/persistence/data_persistence.c

OBJECTS = $(OBJ_DIR)/main.o \
          $(OBJ_DIR)/lum/lum_core.o \
          $(OBJ_DIR)/vorax/vorax_operations.o \
          $(OBJ_DIR)/parser/vorax_parser.o \
          $(OBJ_DIR)/binary/binary_lum_converter.o \
          $(OBJ_DIR)/logger/lum_logger.o \
          $(OBJ_DIR)/optimization/memory_optimizer.o \
          $(OBJ_DIR)/optimization/pareto_optimizer.o \
          $(OBJ_DIR)/optimization/pareto_inverse_optimizer.o \
          $(OBJ_DIR)/parallel/parallel_processor.o \
          $(OBJ_DIR)/metrics/performance_metrics.o \
          $(OBJ_DIR)/crypto/crypto_validator.o \
          $(OBJ_DIR)/persistence/data_persistence.o

EXECUTABLE = $(BIN_DIR)/lum_vorax

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/lum $(OBJ_DIR)/vorax $(OBJ_DIR)/parser $(OBJ_DIR)/binary $(OBJ_DIR)/logger $(OBJ_DIR)/optimization $(OBJ_DIR)/parallel $(OBJ_DIR)/metrics $(OBJ_DIR)/crypto $(OBJ_DIR)/persistence

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

$(EXECUTABLE): $(OBJECTS) | $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ -lpthread -lm

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/parallel/parallel_processor.o: $(SRC_DIR)/parallel/parallel_processor.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -pthread -c $< -o $@

.PHONY: all clean run test
.DEFAULT_GOAL := all

all: $(EXECUTABLE) | $(LOG_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR) *.o *.log

run: $(EXECUTABLE)
	./$(EXECUTABLE)

test: $(EXECUTABLE)
	./$(EXECUTABLE) --test
