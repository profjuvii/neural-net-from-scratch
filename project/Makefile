CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -std=c99
LIB_DIR = src
OUTPUT = program

all: $(OUTPUT)

$(OUTPUT): main.o braincraft.o optimizers.o activation_funcs.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $^ -o $@

braincraft.o: $(LIB_DIR)/braincraft.c
	$(CC) $(CFLAGS) -c $^ -o $@

optimizers.o: $(LIB_DIR)/optimizers.c
	$(CC) $(CFLAGS) -c $^ -o $@

activation_funcs.o: $(LIB_DIR)/activation_funcs.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -rf *.o $(OUTPUT)
