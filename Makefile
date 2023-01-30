CFLAGS = -O2 -Wall -Wextra -Wpedantic
OBJ = libzerdax.so
SRC = c_filter.c c_segments.c c_lines_bundle.c

all: libzerdax.so

libzerdax.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

clean:
	rm $(OBJ)
