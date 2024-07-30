CC = gcc

PNG_INCLUDE_DIR = /opt/homebrew/Cellar/libpng/1.6.43/include
PNG_LIB_DIR = /opt/homebrew/Cellar/libpng/1.6.43/lib

CFLAGS = -Wall -Wextra -std=c99 -Iinclude -I$(PNG_INCLUDE_DIR)
LDFLAGS = -L$(PNG_LIB_DIR) -lpng

SRC = src
OBJ = main.o utils.o data_loader.o image2vector.o braincraft.o optimizers.o loss_funcs.o activation_funcs.o
OUTPUT = program

all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: $(SRC)/utils.c
	$(CC) $(CFLAGS) -c $< -o $@

data_loader.o: $(SRC)/data_loader.c
	$(CC) $(CFLAGS) -c $< -o $@

image2vector.o: $(SRC)/image2vector.c
	$(CC) $(CFLAGS) -c $< -o $@

braincraft.o: $(SRC)/braincraft.c
	$(CC) $(CFLAGS) -c $< -o $@

optimizers.o: $(SRC)/optimizers.c
	$(CC) $(CFLAGS) -c $< -o $@

loss_funcs.o: $(SRC)/loss_funcs.c
	$(CC) $(CFLAGS) -c $< -o $@

activation_funcs.o: $(SRC)/activation_funcs.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(OUTPUT)
