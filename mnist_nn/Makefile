CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -I../braincraft/include -I/opt/homebrew/Cellar/libpng/1.6.43/include
LDFLAGS = -L../braincraft -lbraincraft -L/opt/homebrew/Cellar/libpng/1.6.43/lib -lpng

OUTPUT = mnist_nn

all: $(OUTPUT)

$(OUTPUT): mnist_nn.c
	$(CC) $(CFLAGS) mnist_nn.c $(LDFLAGS) -o $(OUTPUT)

clean:
	rm -rf *.o $(OUTPUT)
