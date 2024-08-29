CC = gcc
CFLAGS = -I../synapse/1.0.0/include -I/opt/homebrew/Cellar/libpng/1.6.43/include -Wall -Wextra -std=c11
LDFLAGS = -L../synapse/1.0.0/lib -L/opt/homebrew/Cellar/libpng/1.6.43/lib
LIBS = -lsynapse -lpng

TARGET = classifier

$(TARGET): main.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf main.o $(TARGET)
