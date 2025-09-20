CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -D_GNU_SOURCE -D_POSIX_C_SOURCE=199309L -I./src/debug -I./src
LDFLAGS = -pthread -lm -lrt

# Objets de base fonctionnels
CORE_OBJECTS = \
        obj/main.o \
        obj/lum/lum_core.o \
        obj/vorax/vorax_operations.o \
        obj/parser/vorax_parser.o \
        obj/binary/binary_lum_converter.o \
        obj/logger/lum_logger.o \
        obj/logger/log_manager.o \
        obj/debug/memory_tracker.o \
        obj/debug/forensic_logger.o \
        obj/persistence/data_persistence.o

TARGET = bin/lum_vorax

all: $(TARGET)

obj bin:
	mkdir -p obj/lum obj/vorax obj/parser obj/binary obj/logger obj/debug obj/persistence bin

$(TARGET): $(CORE_OBJECTS) | bin
	$(CC) $^ -o $@ $(LDFLAGS)

obj/%.o: src/%.c | obj
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj bin

.PHONY: all clean