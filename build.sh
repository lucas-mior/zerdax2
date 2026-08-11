#!/bin/sh -e

# shellcheck disable=SC2086

dir=$(dirname "$(readlink -f "$0")")
# shellcheck source=/dev/null
. "$dir/cbase/common.sh"

cd "$dir" || exit
script=$(basename "$0")
target="${1:-debug}"

printf "\n${script} ${RED}${1:-} ${2:-}$RES\n"

PREFIX="${PREFIX:-/usr/local}"
DESTDIR="${DESTDIR:-/}"

src="c_filter.c"
lib="bin/libzerdax.so"
cfilter="bin/cfilter"
csegments="bin/csegments"
mkdir -p bin

CC=$(get_compiler "$target")

CPPFLAGS="$CPPFLAGS -I$dir/cbase"

CFLAGS="$CFLAGS -std=c11"
CFLAGS="$CFLAGS -Wfatal-errors"
CFLAGS="$CFLAGS -Wextra -Wall"
CFLAGS="$CFLAGS -Werror=all -Werror=extra"
CFLAGS="$CFLAGS -Werror"  # Only uncomment occasionally, keep this line

if [ "$CC" = "clang" ]; then
    CFLAGS="$CFLAGS -Weverything"
    CFLAGS="$CFLAGS -Wno-assign-enum"
    CFLAGS="$CFLAGS -Wno-c++-keyword"
    CFLAGS="$CFLAGS -Wno-cast-qual"
    CFLAGS="$CFLAGS -Wno-constant-logical-operand"
    CFLAGS="$CFLAGS -Wno-covered-switch-default"
    CFLAGS="$CFLAGS -Wno-disabled-macro-expansion"
    CFLAGS="$CFLAGS -Wno-float-equal"
    CFLAGS="$CFLAGS -Wno-format-nonliteral"
    CFLAGS="$CFLAGS -Wno-implicit-int-enum-cast"
    CFLAGS="$CFLAGS -Wno-implicit-void-ptr-cast"
    CFLAGS="$CFLAGS -Wno-pre-c11-compat"
    CFLAGS="$CFLAGS -Wno-unsafe-buffer-usage"
    CFLAGS="$CFLAGS -Wno-unused-macros"
    CFLAGS="$CFLAGS -Wno-used-but-marked-unused"
fi

LDFLAGS="$LDFLAGS -lm -lpthread"

case "$target" in
debug)
    CFLAGS="$CFLAGS -g3 -Og -fPIC"
    CPPFLAGS="$CPPFLAGS -DDEBUGGING=1"
    ;;
build|all|libzerdax.so|cfilter|csegments)
    CFLAGS="$CFLAGS -g3 -O2 -fPIC -flto -march=native -ftree-vectorize"
    ;;
fast_feedback)
    CFLAGS="$CFLAGS -fPIC"
    ;;
test|install|uninstall|clean)
    ;;
*)
    CFLAGS="$CFLAGS -O2"
    ;;
esac

build_library () {
    build_tags
    trace_on
    $CC $CPPFLAGS $CFLAGS -shared -o "$lib" $LDFLAGS "$src"
    trace_off
}

build_cfilter () {
    trace_on
    $CC -DTESTING_c_filter=1 $CPPFLAGS $CFLAGS -o "$cfilter" "c_filter.c" $LDFLAGS
    trace_off
}

build_csegments () {
    trace_on
    $CC $CPPFLAGS $CFLAGS -o "$csegments" "c_segments.c" $LDFLAGS
    trace_off
}

case "$target" in
clean)
    trace_on
    rm -rf bin tags .tags.vim
    trace_off
    ;;
test)
    TEST_EXCLUDE_PATTERN='(^|/)cbase/' test "$2"
    exit
    ;;
check)
    CC=gcc CFLAGS="-fanalyzer -fdiagnostics-color=never" "$0" build
    CFLAGS="--analyze -Xanalyzer -analyzer-output=text"
    CFLAGS="$CFLAGS -Xanalyzer -analyzer-werror"
    CFLAGS="$CFLAGS -Xanalyzer -analyzer-opt-analyze-headers"
    CFLAGS="$CFLAGS -Wno-unused-command-line-argument"
    CFLAGS="$CFLAGS -fno-color-diagnostics"
    CC=clang CFLAGS="$CFLAGS" "$0" build
    exit
    ;;
install)
    if [ ! -f "$lib" ] || [ ! -f "$cfilter" ] || [ ! -f "$csegments" ]; then
        "$0" build
    fi
    trace_on
    install -Dm755 "$lib" "${DESTDIR}${PREFIX}/lib/libzerdax.so"
    install -Dm755 "$cfilter" "${DESTDIR}${PREFIX}/bin/cfilter"
    install -Dm755 "$csegments" "${DESTDIR}${PREFIX}/bin/csegments"
    trace_off
    ;;
uninstall)
    trace_on
    rm -f "${DESTDIR}${PREFIX}/lib/libzerdax.so"
    rm -f "${DESTDIR}${PREFIX}/bin/cfilter"
    rm -f "${DESTDIR}${PREFIX}/bin/csegments"
    trace_off
    ;;
libzerdax.so)
    build_library
    ;;
cfilter)
    build_cfilter
    ;;
csegments)
    build_csegments
    ;;
*)
    build_library
    build_cfilter
    build_csegments
    ;;
esac
