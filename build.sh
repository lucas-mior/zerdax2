#!/bin/sh -e

# shellcheck disable=SC2086

dir=$(dirname "$(readlink -f "$0")")
# shellcheck source=/dev/null
. "$dir/cbase/common.sh"

CPPFLAGS="$CPPFLAGS -I$dir/cbase"
cd "$dir" || exit
program=$(basename "$(readlink -f "$(dirname "$0")")")
script=$(basename "$0")
target="${1:-debug}"

if [ "$target" = "test" ]; then
    exit
fi

printf "
${script} ${RED}${1:-} ${2:-}$RES
"

PREFIX="${PREFIX:-/usr/local}"
DESTDIR="${DESTDIR:-/}"

src="c_filter.c"
lib="bin/libzerdax.so"
cfilter="bin/cfilter"
csegments="bin/csegments"
mkdir -p bin

CPPFLAGS="$CPPFLAGS -D_DEFAULT_SOURCE"
CFLAGS="$CFLAGS -std=c11"
CFLAGS="$CFLAGS -Wfatal-errors"
CFLAGS="$CFLAGS -Wextra -Wall"
CFLAGS="$CFLAGS -Werror"
CFLAGS="$CFLAGS -Wno-format-pedantic"
CFLAGS="$CFLAGS -Wno-unknown-warning-option"
CFLAGS="$CFLAGS -Wno-gnu-union-cast"
CFLAGS="$CFLAGS -Wno-unused-macros"
CFLAGS="$CFLAGS -Wno-constant-logical-operand"
CFLAGS="$CFLAGS -Wno-float-equal"
CFLAGS="$CFLAGS -Wno-undefined-internal"
CFLAGS="$CFLAGS -Wno-cast-qual"
CFLAGS="$CFLAGS -Wno-unknown-pragmas"
CPPFLAGS="$CPPFLAGS -D_XOPEN_SOURCE=700"
CFLAGS="$CFLAGS -Wno-implicit-void-ptr-cast"
LDFLAGS="$LDFLAGS -lm -lpthread"

OS=$(uname -a)
GNUSOURCE=
if echo "$OS" | grep -q "Linux"; then
    if echo "$OS" | grep -q "GNU"; then
        GNUSOURCE="-D_GNU_SOURCE"
    fi
fi

case "$target" in
debug|test)
    CC="${CC:-tcc}"
    ;;
fast_feedback)
    CC="${CC:-clang}"
    ;;
*)
    CC="${CC:-cc}"
    ;;
esac

if ! command -v "$CC" > /dev/null 2>&1; then
    CC=cc
fi

if [ "$CC" = "clang" ]; then
    CFLAGS="$CFLAGS -Weverything"
    CFLAGS="$CFLAGS -Wno-unsafe-buffer-usage"
    CFLAGS="$CFLAGS -Wno-format-nonliteral"
    CFLAGS="$CFLAGS -Wno-disabled-macro-expansion"
    CFLAGS="$CFLAGS -Wno-c++-keyword"
    CFLAGS="$CFLAGS -Wno-pre-c11-compat"
    CFLAGS="$CFLAGS -Wno-implicit-void-ptr-cast"
    CFLAGS="$CFLAGS -Wno-ignored-attributes"
    CFLAGS="$CFLAGS -Wno-covered-switch-default"
    CFLAGS="$CFLAGS -Wno-used-but-marked-unused"
    CFLAGS="$CFLAGS -Wno-implicit-int-enum-cast"
    CFLAGS="$CFLAGS -Wno-assign-enum"
    CFLAGS="$CFLAGS -Wno-cast-function-type-strict"
    CFLAGS="$CFLAGS -Wno-bad-function-cast"
fi
case "$target" in
debug)
    CFLAGS="$CFLAGS -g3 -O0 -fPIC"
    CPPFLAGS="$CPPFLAGS $GNUSOURCE -DDEBUGGING=1"
    ;;
build|all|libzerdax.so|cfilter|csegments)
    CFLAGS="$CFLAGS $GNUSOURCE -g3 -O2 -fPIC -flto -march=native -ftree-vectorize"
    ;;
fast_feedback)
    CFLAGS="$CFLAGS $GNUSOURCE -fPIC -Werror"
    ;;
test|install|uninstall|clean)
    ;;
*)
    CFLAGS="$CFLAGS -O2"
    ;;
esac

build_tags () {
    if command -v ctags >/dev/null 2>&1; then
        find . -iname "*.[ch]" -print0             | xargs -0 ctags --kinds-C=+l+d 2> /dev/null || true
    fi

    if [ -f tags ] && command -v vtags.sed >/dev/null 2>&1; then
        vtags.sed tags | sort | uniq > .tags.vim 2> /dev/null || true
    fi
}

install_opt () {
    mode="$1"
    file="$2"
    dest="$3"

    if [ -f "$file" ]; then
        install "$mode" "$file" "$dest"
    elif [ -d "$file" ]; then
        install "$mode" "$dest"
        cp -rp "$file/." "$dest/"
    fi
}

uninstall_opt () {
    file="$1"
    dest="$2"

    if [ -e "$file" ]; then
        rm -rf "$dest"
    fi
}
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
