CC = gcc
CFLAGS = -std=c11 -Wall -Werror
OUTPUT = program

all: $(OUTPUT)

$(OUTPUT): main.o braincraft.o activation_funcs.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $^ -o $@

braincraft.o: braincraft.c
	$(CC) $(CFLAGS) -c $^ -o $@

activation_funcs.o: activation_funcs.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -rf *.o $(OUTPUT)
