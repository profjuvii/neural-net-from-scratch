CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -std=c99
LIB_SRC = src
OUTPUT = program

all: $(OUTPUT)

$(OUTPUT): main.o braincraft.o optimizers.o activation_funcs.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

braincraft.o: $(LIB_SRC)/braincraft.c
	$(CC) $(CFLAGS) -c $< -o $@

optimizers.o: $(LIB_SRC)/optimizers.c
	$(CC) $(CFLAGS) -c $< -o $@

activation_funcs.o: $(LIB_SRC)/activation_funcs.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(OUTPUT)
