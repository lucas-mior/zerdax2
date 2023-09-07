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

typedef int32_t int32;
static const int32 WW = 512;
typedef struct Slice {
    int32 start_y;
    int32 end_y;
} Slice;

static double *restrict input;
static double *restrict weights;
static double *restrict normalization;
static double *restrict output;
static int32 hh;

void filter(double *restrict, double *restrict , 
            double *restrict, double *restrict,
            int32 const);
static void matrix_weights(void);
static void matrix_normalization(void);
static void matrix_convolute(void);
static int weights_slice(void *);
static inline double weight(int32, int32);

void filter(double *restrict input0, double *restrict output0, 
            double *restrict normalization0, double *restrict weights0,
            int32 const hh0) {

    input = input0;
    weights = weights0;
    normalization = normalization0;
    output = output0;
    hh = hh0;

    matrix_weights();
    matrix_normalization();
    matrix_convolute();
}

void matrix_weights(void) {
    memset(weights, 0, WW*hh*sizeof (*weights));

    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (number_threads > 8)
        number_threads = 8;
    int32 range = (hh - 2) / number_threads;
    
    thrd_t threads[number_threads];
    Slice *slice = malloc(number_threads * sizeof (*slice));
    if (slice == NULL) {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < number_threads; i += 1) {
        slice[i].start_y = i*range + 1;
        if (i == number_threads - 1) {
            slice[i].end_y = hh - 1;
        } else {
            slice[i].end_y = (i + 1)*range + 1;
        }

        thrd_create(&threads[i], weights_slice, (void *) &slice[i]);
    }

    for (int i = 0; i < number_threads; i += 1) {
        thrd_join(threads[i], NULL);
    }
}

int weights_slice(void *arg) {
    Slice *slice = arg;

    for (int32 y = slice->start_y; y < slice->end_y; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            weights[WW*y + x] = weight(x, y);
        }
    }

    thrd_exit(0);
}

double weight(int32 x, int32 y) {
    double Gx, Gy;
    double d, w;

    Gx = input[WW*y + x+1] - input[WW*y + x-1];
    Gy = input[WW*(y+1) + x] - input[WW*(y-1) + x];

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void matrix_normalization(void) {
    memset(normalization, 0, WW*hh*sizeof (*normalization));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < WW - 1; x += 1) {
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    normalization[WW*y + x] += weights[WW*(y+i) + x+j];
                }
            }
        }
    }
}

void matrix_convolute(void) {
    memset(output, 0, WW*hh*sizeof (*output));
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
    for (int32 x = 0; x < (WW*hh - 1); x += WW)
        output[x] = output[x+1];
    for (int32 y = 0; y < WW-1; y += 1)
        output[y] = output[y+WW];
    for (int32 x = WW-1; x < (WW*hh - 1); x += WW)
        output[x] = output[x-1];
    for (int32 y = (hh-1)*WW; y < (WW*hh - 1); y += 1)
        output[y] = output[y-WW];
}
