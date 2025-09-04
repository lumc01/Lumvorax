CC = clang
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SOURCES = $(SRCDIR)/main.c \
          $(SRCDIR)/lum/lum_core.c \
          $(SRCDIR)/vorax/vorax_operations.c \
          $(SRCDIR)/parser/vorax_parser.c \
          $(SRCDIR)/binary/binary_lum_converter.c \
          $(SRCDIR)/logger/lum_logger.c

# Object files
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Executable
EXECUTABLE = $(BINDIR)/lum_vorax

# Create directories
$(OBJDIR):
	mkdir -p $(OBJDIR)/lum $(OBJDIR)/vorax $(OBJDIR)/parser $(OBJDIR)/binary $(OBJDIR)/logger

$(BINDIR):
	mkdir -p $(BINDIR)

# Build executable
$(EXECUTABLE): $(OBJECTS) | $(BINDIR)
	$(CC) $(OBJECTS) -o $@

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Targets
.PHONY: all clean run test install

all: $(EXECUTABLE)

clean:
	rm -rf $(OBJDIR) $(BINDIR) *.log

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