#!/bin/sh -e

# shellcheck disable=SC2086

dir=$(dirname "$(readlink -f "$0")")
cd "$dir" || exit

# shellcheck source=./cbase/common.sh
. "./cbase/common.sh"

script=$(basename "$0")
common_build_parse_args "$@"

case "$mode" in
all|build|cfilter|check|clean|cross|csegments|debug|fast_feedback|install|libzerdax.so|test|uninstall)
    ;;
*)
    common_build_unknown_mode
    ;;
esac

common_build_print_invocation "$script"

PREFIX="${PREFIX:-/usr/local}"
DESTDIR="${DESTDIR:-/}"

src="c_filter.c"
lib="bin/libzerdax.so"
libflag="-shared"
cfilter="bin/cfilter"
csegments="bin/csegments"
mkdir -p bin

CC=$(common_get_compiler "$mode")

CPPFLAGS="$CPPFLAGS -I$dir/cbase"

CFLAGS="$CFLAGS -std=c11"
CFLAGS="$CFLAGS -Wfatal-errors"
CFLAGS="$CFLAGS -Wextra -Wall"
CFLAGS="$CFLAGS -Werror=all -Werror=extra"
# CFLAGS="$CFLAGS -Werror"  # Only uncomment occasionally, keep this line

if [ "$CC" = "clang" ] || [ "$CC" = "zig cc" ]; then
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

LDFLAGS="$LDFLAGS -lm"

case "$mode" in
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
cross)
    common_build_cross_all windows
    cross="$target"

    CFLAGS="$CFLAGS -O2"
    CFLAGS="$CFLAGS -fPIC"
    CFLAGS="$CFLAGS -Wno-padded"
    CFLAGS="$CFLAGS -target $cross"

    case "$cross" in
    *macos*)
        lib="bin/libzerdax.dylib"
        libflag="-dynamiclib"
        ;;
    *)
        ;;
    esac
    ;;
test|install|uninstall|clean)
    ;;
all|build|cfilter|check|clean|cross|csegments|debug|fast_feedback|install|libzerdax.so|test|uninstall)
    ;;
*)
    common_build_unknown_mode
    ;;
esac

build_library () {
    common_build_tags
    trace_on
    $CC $CPPFLAGS $CFLAGS $libflag -o "$lib" $LDFLAGS "$src"
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

case "$mode" in
clean)
    trace_on
    rm -rf bin tags .tags.vim
    trace_off
    ;;
test)
    TEST_EXCLUDE_PATTERN='(^|/)cbase/' common_test "$target"
    exit
    ;;
check)
    common_build_run_analyzers build
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
all|build|cross|debug|fast_feedback)
    build_library
    build_cfilter
    build_csegments
    ;;
esac
