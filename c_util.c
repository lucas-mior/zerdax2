#ifndef C_UTIL_C
#define C_UTIL_C

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>

static void
error(char *format, ...) {
    int n;
    ssize_t w;
    va_list args;
    char buffer[BUFSIZ];

    va_start(args, format);
    n = vsnprintf(buffer, sizeof(buffer) - 1, format, args);
    va_end(args);

    if (n < 0) {
        fprintf(stderr, "Error in vsnprintf()\n");
        exit(EXIT_FAILURE);
    }

    buffer[n] = '\0';
    if ((w = write(STDERR_FILENO, buffer, (size_t)n)) < n) {
        fprintf(stderr, "Error writing to STDERR_FILENO");
        if (w < 0)
            fprintf(stderr, ": %s", strerror(errno));
        fprintf(stderr, ".\n");
    }
}

static void *
util_malloc(size_t size) {
    void *p;
    if ((p = malloc(size)) == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes.\n", size);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *
util_realloc(void *old, size_t size) {
    void *p;
    if ((p = realloc(old, size)) == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes.\n", size);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *
util_calloc(size_t nmemb, size_t size) {
    void *p;
    if ((p = calloc(nmemb, size)) == NULL) {
        fprintf(stderr, "Failed to allocate %zu members of %zu bytes each.\n", 
                        nmemb, size);
        exit(EXIT_FAILURE);
    }
    return p;
}

#endif
