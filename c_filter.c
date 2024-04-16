/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <immintrin.h>
#include <threads.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "c_declarations.h"

#define WW0 512
#define NTHREADS 8

typedef int32_t int32;
typedef uint32_t uint32;
typedef float floaty;

static const int32 WW = WW0;

static floaty *restrict input;
static floaty *restrict weights;
static floaty *restrict output;
static int32 hh;
static uint32 matrix_size;

void filter(floaty *restrict, floaty *restrict,
            floaty *restrict, int32 const);

typedef struct Slice {
    uint32 y0;
    uint32 y1;
    uint32 id;
} Slice;

int
work(void *arg) {
    Slice *slice = arg;
    uint32 y0 = slice->y0;
    uint32 y1 = slice->y1;
    uint32 id = slice->id;

    for (uint32 y = y0; y < (uint32) (y1 + 1); y += 1) {
        for (uint32 x = 1; x < WW - 1; x += 1) {
            floaty Gx, Gy;
            floaty d, w;

            Gx = input[WW*y + x+1] - input[WW*y + x-1];
            Gy = input[WW*(y+1) + x] - input[WW*(y-1) + x];

            d = sqrtf(Gx*Gx + Gy*Gy);
            w = expf(-sqrtf(d));
            weights[WW*y + x] = w;
        }
    }

    usleep(100);

    for (int32 y = y0; y < (int32) y1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            floaty norm = 0;
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    floaty w = weights[WW*(y+i) + x+j];
                    norm += w;
                    output[WW*y + x] += w*input[WW*(y+i) + x+j];
                }
            }
            output[WW*y + x] /= norm;
        }
    }

    thrd_exit(0);
}

void
filter(floaty *restrict input0, floaty *restrict output0,
       floaty *restrict weights0, int32 const hh0) {

    input = input0;
    weights = weights0;
    output = output0;
    hh = hh0;
    matrix_size = (uint32) WW * (uint32) hh;

    memset(weights, 0, (size_t) matrix_size * sizeof (*weights));
    memset(output, 0, matrix_size * sizeof (*output));

    thrd_t threads[NTHREADS];
    Slice slices[NTHREADS];
    uint32 range = hh / NTHREADS;

    for (uint32 i = 0; i < (NTHREADS - 1); i += 1) {
        slices[i].y0 = i*range + 1;
        slices[i].y1 = (i+1)*range + 1;
        slices[i].id = i;

        thrd_create(&threads[i], work, (void *) &slices[i]);
    }{
        uint32 i = NTHREADS - 1;
        slices[i].y0 = i*range + 1;
        slices[i].y1 = hh - 1;
        slices[i].id = i;

        thrd_create(&threads[i], work, (void *) &slices[i]);
    }

    for (uint32 i = 0; i < NTHREADS; i += 1) {
        thrd_join(threads[i], NULL);
    }

    for (uint32 x = 0; x < (matrix_size - 1); x += WW)
        output[x] = output[x+1];
    for (uint32 y = 0; y < WW-1; y += 1)
        output[y] = output[y+WW];
    for (uint32 x = WW-1; x < (matrix_size - 1); x += WW)
        output[x] = output[x-1];
    for (uint32 y = (uint32)(hh-1)*WW; y < (matrix_size - 1); y += 1)
        output[y] = output[y-WW];

    return;
}

#ifndef TESTING_THIS_FILE
#define TESTING_THIS_FILE 0
#endif

#if TESTING_THIS_FILE
#define HH0 512
#define IMAGE_SIZE HH0*WW0
static long hash(floaty *array) {
    long hash = 5381;
    for (int i = 0; i < IMAGE_SIZE; i += 1) {
        long c;
        memcpy(&c, &array[i], sizeof (c));
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static floaty randd(void) {
    uint r53 = ((uint) rand() << 10) ^ ((uint) rand() >> 1);
    return (floaty) r53 / 9007199.0f; // 2^53 - 1
}

int main(int argc, char **argv) {
    int hh0 = HH0;
    int nfilters = 1000;

    floaty *input0 = malloc(IMAGE_SIZE*sizeof(floaty));
    floaty *output0 = malloc(IMAGE_SIZE*sizeof(floaty));
    floaty *weights0 = malloc(IMAGE_SIZE*sizeof(floaty));

    struct timespec t0, t1;
    (void) argc;
    (void) argv;

    for (int i = 0; i < IMAGE_SIZE; i += 1) {
        input0[i] = randd();
    }

    printf("input0: %ld\n", hash(input0));
    clock_gettime(CLOCK_REALTIME, &t0);
    
    for (int i = 0; i < nfilters; i += 1) {
        filter(input0, output0, weights0, hh0);
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("output0: %ld\n", hash(output0));

    {
        long diffsec = t1.tv_sec - t0.tv_sec;
        long diffnsec = t1.tv_nsec - t0.tv_nsec;
        floaty time_elapsed = (floaty) diffsec + (floaty) diffnsec/1.0e9f;
        printf("time elapsed for %dx%d: %f [%f / filter]\n",
                WW0, HH0, time_elapsed, time_elapsed / nfilters);
    }

    {
        uint8 *gray = malloc(IMAGE_SIZE*sizeof(*gray));
        FILE *image1 = fopen("input.data", "w");
        FILE *image2 = fopen("output.data", "w");

        for (int i = 0; i < IMAGE_SIZE; i += 4) {
            gray[i+0] = (uint8) output0[i+0]*255;
            gray[i+1] = (uint8) output0[i+1]*255;
            gray[i+2] = (uint8) output0[i+2]*255;
            gray[i+3] = (uint8) output0[i+3]*255;
        }

        fwrite(input0, sizeof (*input0), IMAGE_SIZE, image1);
        fwrite(gray, sizeof (*gray), IMAGE_SIZE, image2);

        free(gray);
    }
    free(input0);
    free(output0);
    free(weights0);

    return 0;
}
#endif
