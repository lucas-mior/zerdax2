#!/bin/sh
set -e

CC="clang"

CFLAGS="-g -O3 -march=native -fPIC -flto -D_DEFAULT_SOURCE"
CFLAGS="$CFLAGS -Wall -Wextra -Wno-unsafe-buffer-usage -Wno-unused-macros -Wno-unused-function"
CFLAGS="$CFLAGS -Wno-implicit-void-ptr-cast"
CFLAGS="$CFLAGS -Weverything -Wno-format-nonliteral"

CPPFLAGS="-Icbase"
LDFLAGS="-lm -lpthread"

SRC="c_filter.c"

TARGET="${1:-all}"

if [ "$TARGET" = "clean" ]; then
    rm -f libzerdax.so cfilter
    exit 0
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "libzerdax.so" ]; then
    ctags --kinds-C=+l *.h *.c || true
    vtags.sed tags > .tags.vim || true
    $CC $CPPFLAGS $CFLAGS -o libzerdax.so $LDFLAGS $SRC -shared
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "cfilter" ]; then
    $CC $CPPFLAGS $CFLAGS -o cfilter $LDFLAGS c_filter.c
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "csegments" ]; then
    $CC $CPPFLAGS $CFLAGS -o csegments $LDFLAGS c_segments.c
fi
