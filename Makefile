CC = clang
CFLAGS = -g -O3 -fPIC -fopenmp
CFLAGS += -Weverything -Wall -Wextra -Wno-unsafe-buffer-usage 
LDFLAGS = -lm
OBJ = libzerdax.so
SRC = c_filter.c c_segments.c c_lines_bundle.c c_util.c

all: libzerdax.so cfilter

libzerdax.so: $(SRC) Makefile
	-ctags --kinds-C=+l *.h *.c
	-vtags.sed tags > .tags.vim
	$(CC) $(CFLAGS) -shared -o $(OBJ) $(LDFLAGS) $(SRC)

cfilter: $(SRC) Makefile
	$(CC) $(CFLAGS) -o cfilter $(LDFLAGS) $(SRC) -DTESTING_THIS_FILE=1

clean:
	rm $(OBJ) cfilter
