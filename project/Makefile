CC = gcc
CFLAGS = -Iinclude -I/opt/homebrew/Cellar/libpng/1.6.43/include -Wall -Wextra -std=c99
LIB_DIR = /opt/homebrew/Cellar/libpng/1.6.43/lib
LIBS = -L$(LIB_DIR) -lpng
LIB_SRC = src
OUTPUT = program

all: $(OUTPUT)

$(OUTPUT): main.o braincraft.o optimizers.o activation_funcs.o img_vec.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

img_vec.o: $(LIB_SRC)/img_vec.c
	$(CC) $(CFLAGS) -c $< -o $@

braincraft.o: $(LIB_SRC)/braincraft.c
	$(CC) $(CFLAGS) -c $< -o $@

optimizers.o: $(LIB_SRC)/optimizers.c
	$(CC) $(CFLAGS) -c $< -o $@

activation_funcs.o: $(LIB_SRC)/activation_funcs.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(OUTPUT)
