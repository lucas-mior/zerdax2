#!/bin/sh

# shellcheck disable=SC2086
set -e

CC="${CC:-cc}"

CFLAGS="-std=c11 -D_DEFAULT_SOURCE"
CFLAGS="$CFLAGS -g -O3 -march=native -fPIC -flto"
CFLAGS="$CFLAGS -Wfatal-errors"
CFLAGS="$CFLAGS -Werror"
CFLAGS="$CFLAGS -Wall -Wextra"

CFLAGS="$CFLAGS -Wno-unused-function"
CFLAGS="$CFLAGS -Wno-unused-macros"
CFLAGS="$CFLAGS -Wno-implicit-void-ptr-cast"

if [ $CC = "clang" ]; then
    CFLAGS="$CFLAGS -Weverything"
    CFLAGS="$CFLAGS -Wno-format-nonliteral"
    CFLAGS="$CFLAGS -Wno-unsafe-buffer-usage"
    CFLAGS="$CFLAGS -Wno-covered-switch-default"
    CFLAGS="$CFLAGS -Wno-c++-keyword"
    CFLAGS="$CFLAGS -Wno-pre-c11-compat"
    CFLAGS="$CFLAGS -Wno-constant-logical-operand"
    CFLAGS="$CFLAGS -Wno-cast-qual"

    # TODO: implement safe floating point comparisons
    CFLAGS="$CFLAGS -Wno-float-equal"
fi

dir=$(dirname "$(readlink -f "$0")")
cbase="cbase"
CPPFLAGS="$CPPFLAGS -I$dir/$cbase"
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
