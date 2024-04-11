/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>
#include "c_declarations.h"

#define WW0 512

typedef int32_t int32;
typedef uint32_t uint32;
typedef float floaty;

static const int32 WW = WW0;

static floaty *restrict input;
static floaty *restrict weights;
static floaty *restrict normalization;
static floaty *restrict output;
static int32 hh;
static uint32 matrix_size;
static long number_threads = 0;

void filter(floaty *restrict, floaty *restrict,
            floaty *restrict, floaty *restrict,
            int32 const);
static void matrix_weights(void);
static void matrix_convolute(void);
static int weights_slice(void *);

void
filter(floaty *restrict input0, floaty *restrict output0,
       floaty *restrict normalization0, floaty *restrict weights0,
       int32 const hh0) {

    number_threads = sysconf(_SC_NPROCESSORS_ONLN);

    input = input0;
    weights = weights0;
    normalization = normalization0;
    output = output0;
    hh = hh0;
    matrix_size = (uint32) WW * (uint32) hh;

    matrix_weights();
    matrix_convolute();
    return;
}

typedef struct Slice {
    uint32 start_y;
    uint32 end_y;
} Slice;

void
matrix_weights(void) {
    uint32 nthreads;
    uint32 range;
    thrd_t *threads;
    Slice *slices;

    if (number_threads > 8) {
        nthreads = 8;
    } else if (number_threads < 1) {
        nthreads = 1;
    } else {
        nthreads = (uint32) number_threads;
    }

    range = (uint32) (hh - 2) / nthreads;
    threads = util_malloc(nthreads * sizeof (*threads));
    slices = util_malloc(nthreads * sizeof (*slices));

    memset(weights, 0, (size_t) matrix_size * sizeof (*weights));
    for (uint32 i = 0; i < nthreads; i += 1) {
        slices[i].start_y = i*range + 1;
        if (i == nthreads - 1) {
            slices[i].end_y = (uint32) hh - 1;
        } else {
            slices[i].end_y = (uint32) (i + 1)*range + 1;
        }

        thrd_create(&threads[i], weights_slice, (void *) &slices[i]);
    }

    for (uint32 i = 0; i < nthreads; i += 1) {
        thrd_join(threads[i], NULL);
    }
    free(threads);
    free(slices);
    return;
}

int
weights_slice(void *arg) {
    Slice *slice = arg;

    for (uint32 y = slice->start_y; y < slice->end_y; y += 1) {
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

    thrd_exit(0);
}

void
matrix_convolute(void) {
    memset(output, 0, matrix_size * sizeof (*output));
    for (int32 y = 1; y < hh - 1; y += 1) {
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
    floaty *normalization0 = malloc(IMAGE_SIZE*sizeof(floaty));
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
        filter(input0, output0, normalization0, weights0, hh0);
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
    free(normalization0);
    free(weights0);

    return 0;
}
#endif
