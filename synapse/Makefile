CC = gcc
CFLAGS = -Iinclude -I/opt/homebrew/Cellar/libpng/1.6.43/include -Wall -Wextra -std=c11

SRC = $(wildcard src/*.c)
OBJ = $(SRC:src/%.c=obj/%.o)

TARGET = libsynapse.a

all: obj lib/$(TARGET) clean

lib/$(TARGET): $(OBJ)
	ar rcs $@ $^

obj/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@
	
obj:
	mkdir -p obj

clean:
	rm -rf obj
