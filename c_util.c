#ifndef C_UTIL_C
#define C_UTIL_C

#include <stdio.h>
#include <stdlib.h>

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
