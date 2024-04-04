/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <time.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <unistd.h>
#include <math.h>
#include <immintrin.h>
#include "c_declarations.h"

/* #define THREADS sysconf(_SC_NPROCESSORS_ONLN) */
#define THREADS 16
#define VSIZE 4
#define HH0 512
#define WW0 512
#define IMAGE_SIZE HH0*WW0

typedef int32_t int32;
typedef uint32_t uint32;

static const int32 WW = WW0;

static double *restrict input;
static double *restrict weights;
static double *restrict normalization;
static double *restrict output;
static int32 hh;
static uint32 matrix_size;

void filter(double *restrict, double *restrict,
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
    long number_threads = THREADS;
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

    __m128d vec0, vec1, vecdiff, vecgrad;

    vec0 = _mm_load_pd(i0);
    vec1 = _mm_load_pd(i1);
    vecdiff = _mm_sub_pd(vec0, vec1);
    vecgrad = _mm_mul_pd(vecdiff, vecdiff);

    _mm_store_pd(G, vecgrad);

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
            __m256d vecn, vec1;
            double n[VSIZE] = {0};
            vecn = _mm256_load_pd(n);

            for (int32 i = -1; i <= +1; i += 1) {
                double w[4];
                memcpy(w, &weights[WW*(y+i) + x-1], sizeof (w));
                vec1 = _mm256_load_pd(w);
                vecn = _mm256_add_pd(vecn, vec1);
            }
            _mm256_store_pd(n, vecn);
            normalization[WW*y + x] = n[0] + n[1] + n[2];
        }
    }
    return;
}

void
matrix_convolute(void) {
    memset(output, 0, matrix_size * sizeof (*output));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            __m256d veco, vecw, veci, vecm;
            double o[VSIZE] = {0};
            veco = _mm256_load_pd(o);

            for (int32 i = -1; i <= +1; i += 1) {
                double weight4[4];
                double input4[4];

                memcpy(weight4, &weights[WW*(y+i) + x-1], sizeof(weight4));
                memcpy(input4, &input[WW*(y+i) + x-1], sizeof(input4));

                vecw = _mm256_load_pd(weight4);
                veci = _mm256_load_pd(input4);
                vecm = _mm256_mul_pd(vecw, veci);
                veco = _mm256_add_pd(veco, vecm);
            }
            _mm256_store_pd(o, veco);
            output[WW*y + x] = o[0] + o[1] + o[2];
        }
    }
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += VSIZE) {
            __m256d veco, vecn;
            double n[VSIZE];
            double o[VSIZE];
            memcpy(n, &normalization[WW*y + x], sizeof (n));
            memcpy(o, &output[WW*y + x], sizeof (n));

            vecn = _mm256_load_pd(n);
            veco = _mm256_load_pd(o);
            veco = _mm256_div_pd(veco, vecn);

            _mm256_store_pd(o, veco);
            memcpy(&output[WW*y + x], o, sizeof (o));
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

static long hash(double *array) {
    long hash = 5381;
    for (int i = 0; i < IMAGE_SIZE; i += 1) {
        long c;
        memcpy(&c, &array[i], sizeof (c));
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static double randd(void) {
    ulong r53 = ((ulong) rand() << 21) ^ ((ulong) rand() >> 2);
    return (double) r53 / 9007199254740991.0; // 2^53 - 1
}

int main(int argc, char **argv) {
    int hh0 = HH0;

    double *input0 = malloc(IMAGE_SIZE*sizeof(double));
    double *output0 = malloc(IMAGE_SIZE*sizeof(double));
    double *normalization0 = malloc(IMAGE_SIZE*sizeof(double));
    double *weights0 = malloc(IMAGE_SIZE*sizeof(double));

    struct timespec t0, t1;
    (void) argc;
    (void) argv;

    for (int i = 0; i < IMAGE_SIZE; i += 1) {
        input0[i] = randd();
    }

    printf("input0: %ld\n", hash(input0));
    clock_gettime(CLOCK_REALTIME, &t0);
    filter(input0, output0, normalization0, weights0, hh0);
    clock_gettime(CLOCK_REALTIME, &t1);
    printf("output0: %ld\n", hash(output0));

    {
        long diffsec = t1.tv_sec - t0.tv_sec;
        long diffnsec = t1.tv_nsec - t0.tv_nsec;
        double time_elapsed = (double) diffsec + (double) diffnsec/1.0e9;
        printf("time elapsed for %d: %f\n", IMAGE_SIZE, time_elapsed);
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
