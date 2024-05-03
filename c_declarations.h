#ifndef C_DECLARATIONS_H
#define C_DECLARATIONS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

typedef int32_t int32;
typedef uint8_t uint8;
typedef unsigned long ulong;

int32
segments_distance(int32 *restrict line0, int32 *restrict line1);

#endif
