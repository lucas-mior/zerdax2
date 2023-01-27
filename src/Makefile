CFLAGS = -O2 -Wall -Wextra -Wpedantic
OBJ = libzerdax.so
SRC = filter.c segments.c

all: libffilter.so

libffilter.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

clean:
	rm $(OBJ)
