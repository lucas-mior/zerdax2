#!/bin/sh

# shellcheck disable=SC2086
set -e

error () {
    >&2 printf "$@"
    return
}

if [ -n "$BASH_VERSION" ]; then
    # shellcheck disable=SC3044
    shopt -s expand_aliases
fi

alias trace_on='set -x'
alias trace_off='{ set +x; } 2>/dev/null'

CC="${CC:-cc}"

CFLAGS="$CFLAGS -std=c11 -D_DEFAULT_SOURCE"
CFLAGS="$CFLAGS -g -O3 -march=native -fPIC -flto"
CFLAGS="$CFLAGS -Wfatal-errors"
# CFLAGS="$CFLAGS -Werror"
CFLAGS="$CFLAGS -Wall -Wextra"
CFLAGS="$CFLAGS -Wno-gnu-union-cast"

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
if [ "$TARGET" = "check" ]; then
    CC=gcc CFLAGS="-fanalyzer" ./build.sh
    scan-build --view -analyze-headers --status-bugs ./build.sh
    exit
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "libzerdax.so" ]; then
    trace_on
    ctags --kinds-C=+l+d cbase/*.c *.h *.c  2> /dev/null || true
    vtags.sed tags | sort | uniq > .tags.vim 2> /dev/null || true
    $CC $CPPFLAGS $CFLAGS -shared -o libzerdax.so $LDFLAGS $SRC
    trace_off
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "cfilter" ]; then
    trace_on
    $CC $CPPFLAGS $CFLAGS -o cfilter $LDFLAGS c_filter.c
    trace_off
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "csegments" ]; then
    trace_on
    $CC $CPPFLAGS $CFLAGS -o csegments $LDFLAGS c_segments.c
    trace_off
fi
