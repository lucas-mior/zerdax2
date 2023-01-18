CFLAGS = -O2 -Wall -Wextra -Wpedantic
OBJ = libffilter.so
SRC = libffilter.c

all: libffilter.so

libffilter.so: $(SRC)
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

clean:
	rm $(OBJ)
