CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -Iinclude -I/opt/homebrew/Cellar/libpng/1.6.43/include
LDFLAGS = -L/opt/homebrew/Cellar/libpng/1.6.43/lib -lpng

SRC_DIR = src
OBJ_DIR = obj
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC))

OUTPUT = libbraincraft.a

all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	ar rcs $@ $(OBJ)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(OUTPUT)
