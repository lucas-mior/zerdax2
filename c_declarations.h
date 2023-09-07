#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

typedef int32_t int32;

int32 segments_distance(int32 * restrict, int32 * restrict);
void *util_malloc(size_t);
void *util_realloc(void *, size_t);
void *util_calloc(size_t, size_t);
