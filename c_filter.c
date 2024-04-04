/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <unistd.h>
#include <math.h>
#include <immintrin.h>
#include "c_declarations.h"

typedef int32_t int32;
typedef uint32_t uint32;

static const int32 WW = 512;

static double *restrict input;
static double *restrict weights;
static double *restrict normalization;
static double *restrict output;
static int32 hh;
static uint32 matrix_size;

void filter(double *restrict, double *restrict , 
            double *restrict, double *restrict,
            int32 const);
static void matrix_weights(void);
static void matrix_normalization(void);
static void matrix_convolute(void);
static int weights_slice(void *);
static inline void weight(double *, double *);
static inline double gradient_sum(uint32 x, uint32 y);

void
filter(double *restrict input0, double *restrict output0, 
       double *restrict normalization0, double *restrict weights0,
       int32 const hh0) {

    input = input0;
    weights = weights0;
    normalization = normalization0;
    output = output0;
    hh = hh0;
    matrix_size = (uint32) WW * (uint32) hh;

    matrix_weights();
    matrix_normalization();
    matrix_convolute();
    return;
}

typedef struct Slice {
    uint32 start_y;
    uint32 end_y;
} Slice;

void
matrix_weights(void) {
    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
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

#define VSIZE 4
int
weights_slice(void *arg) {
    Slice *slice = arg;

    for (uint32 y = slice->start_y; y < slice->end_y; y += 1) {
        for (uint32 x = 1; x < WW - 1; x += VSIZE) {
            double Gsum[VSIZE];
            double w[VSIZE];
            for (uint32 i = 0; i < VSIZE; i += 1) {
                Gsum[i] = gradient_sum(x+i, y);
            }
            weight(Gsum, w);
            memcpy(&weights[WW*y + x], w, sizeof (w));
        }
    }

    thrd_exit(0);
}

double
gradient_sum(uint32 x, uint32 y) {
    double G[2];

    double i0[] = {input[WW*y + x+1], input[WW*(y+1) + x]};
    double i1[] = {input[WW*y + x-1], input[WW*(y-1) + x]};

    __m128d vec0, vec1, vec2;

    vec0 = _mm_load_pd(i0);
    vec1 = _mm_load_pd(i1);
    vec2 = _mm_sub_pd(vec0, vec1);
    vec2 = _mm_mul_pd(vec2, vec2);

    _mm_store_pd(G, vec2); 

    return G[0] + G[1];
}

void
weight(double *Gsum, double *w) {
    __m256d vecd;
    double d[VSIZE];

    
    vecd = _mm256_load_pd(Gsum);
    vecd = _mm256_sqrt_pd(vecd);
    vecd = _mm256_sqrt_pd(vecd);
    _mm256_store_pd(d, vecd);

    for (int i = 0; i < VSIZE; i += 1) {
        w[i] = exp(-d[i]);
    }
    return;
}

void
matrix_normalization(void) {
    memset(normalization, 0, matrix_size * sizeof (*normalization));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    normalization[WW*y + x] += weights[WW*(y+i) + x+j];
                }
            }
        }
    }
    return;
}

void
matrix_convolute(void) {
    memset(output, 0, matrix_size * sizeof (*output));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    output[WW*y + x] += (weights[WW*(y+i) + x+j]*input[WW*(y+i) + x+j]);
                }
            }
            output[WW*y + x] /= normalization[WW*y + x];
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

#define SIZE 262144
static double mean(double *a) {
    double sum = 0;
    for (int i = 0; i < SIZE; i += 1) {
        sum += a[i];
    }
    return sum /= SIZE;
}

int main(int argc, char **argv) {
    int hh0 = 512;

    double *input0 = malloc(SIZE*sizeof(double));
    double *output0 = malloc(SIZE*sizeof(double));
    double *normalization0 = malloc(SIZE*sizeof(double));
    double *weights0 = malloc(SIZE*sizeof(double));

    for (int i = 0; i < SIZE; i += 1) {
        input0[i] = (double)rand() / (double)RAND_MAX;
    }
    
    printf("input0: %f\n", mean(input0));
    filter(input0, output0, normalization0, weights0, hh0);
    printf("output0: %f\n", mean(output0));

    return 0;
}
