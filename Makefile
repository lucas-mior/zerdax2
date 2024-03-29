CC = clang
CFLAGS = -g -O2 -Weverything -Wno-unsafe-buffer-usage
OBJ = libzerdax.so
SRC = c_filter.c c_segments.c c_lines_bundle.c c_util.c

all: libzerdax.so

libzerdax.so: $(SRC) Makefile
	ctags --kinds-C=+l *.h *.c
	vtags.sed tags > .tags.vim
	$(CC) $(CFLAGS) -shared -o $(OBJ) -fPIC -lm $(SRC)

clean:
	rm $(OBJ)
