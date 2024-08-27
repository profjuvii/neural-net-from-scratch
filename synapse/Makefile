CC = gcc
CFLAGS = -Iinclude -I/opt/homebrew/Cellar/libpng/1.6.43/include -Wall -Wextra -std=c11

SRC = $(wildcard src/*.c)
OBJ = $(SRC:src/%.c=obj/%.o)

TARGET = libsynapse.a

all: obj $(TARGET)

obj:
	mkdir -p obj

$(TARGET): $(OBJ)
	ar rcs $@ $^

obj/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj $(TARGET)
