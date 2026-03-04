CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

BIN := hubbard_hts_runner
SRC := src/main.c src/hubbard_hts_module.c

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(BIN)

.PHONY: all clean
