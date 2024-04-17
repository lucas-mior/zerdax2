CC = clang

CFLAGS = -g -O3 -march=native -fPIC -flto -D_DEFAULT_SOURCE
CFLAGS += -Wall -Wextra -Wno-unsafe-buffer-usage -Wno-unused-macros
CFLAGS += -Weverything

LDFLAGS = -lm -lpthread

SRC = c_filter.c c_segments.c c_lines_bundle.c c_util.c

all: libzerdax.so cfilter csegments

libzerdax.so: $(SRC) Makefile
	-ctags --kinds-C=+l *.h *.c
	-vtags.sed tags > .tags.vim
	$(CC) $(CFLAGS) -o libzerdax.so $(LDFLAGS) $(SRC) -shared

cfilter: $(SRC) Makefile
	$(CC) $(CFLAGS) -o cfilter $(LDFLAGS) c_filter.c -DTESTING_THIS_FILE=1

csegments: $(SRC) Makefile
	$(CC) $(CFLAGS) -o csegments $(LDFLAGS) c_segments.c -DTESTING_THIS_FILE=1

clean:
	rm libzerdax.so cfilter
