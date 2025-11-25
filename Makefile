CC = clang

CFLAGS = -g -O3 -march=native -fPIC -flto -D_DEFAULT_SOURCE
CFLAGS += -Wall -Wextra -Wno-unsafe-buffer-usage -Wno-unused-macros -Wno-unused-function
CFLAGS += -Wno-implicit-void-ptr-cast
CFLAGS += -Weverything -Wno-format-nonliteral

LDFLAGS = -lm -lpthread

SRC = c_filter.c

all: libzerdax.so cfilter csegments

libzerdax.so: $(SRC) Makefile
	-ctags --kinds-C=+l *.h *.c
	-vtags.sed tags > .tags.vim
	$(CC) $(CFLAGS) -o libzerdax.so $(LDFLAGS) $(SRC) -shared

cfilter: $(SRC) Makefile
	$(CC) $(CFLAGS) -o cfilter $(LDFLAGS) c_filter.c

csegments: $(SRC) Makefile
	$(CC) $(CFLAGS) -o csegments $(LDFLAGS) c_segments.c

clean:
	rm libzerdax.so cfilter
