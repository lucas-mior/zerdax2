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
static const int32 ww = 512;

static double *restrict input;
static double *restrict weights;
static double *restrict normalization;
static double *restrict output;
static int32 hh;

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

typedef struct ThreadArguments {
    int32 start_y;
    int32 end_y;
} ThreadArguments;

void matrix_weights(void) {
    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (number_threads > 8)
        number_threads = 8;
    int32 range = (hh - 2) / number_threads;
    
    thrd_t threads[number_threads];
    ThreadArguments thread_arguments[number_threads];

    for (int i = 0; i < number_threads; i += 1) {
        thread_arguments[i].start_y = i*range + 1;
        if (i == number_threads - 1) {
            thread_arguments[i].end_y = hh - 1;
        } else {
            thread_arguments[i].end_y = (i + 1)*range + 1;
        }

        thrd_create(&threads[i], weights_slice, (void *) &thread_arguments[i]);
    }

    for (int i = 0; i < number_threads; i += 1) {
        thrd_join(threads[i], NULL);
    }
}

int weights_slice(void *arg) {
    ThreadArguments *args = (ThreadArguments *) arg;

    int32 start_y = args->start_y;
    int32 end_y = args->end_y;

    for (int32 y = start_y; y < end_y; y += 1) {
        for (int32 x = 1; x < ww - 1; x += 1) {
            weights[ww*y + x] = weight(x, y);
        }
    }

    thrd_exit(0);
}

double weight(int32 x, int32 y) {
    double Gx, Gy;
    double d, w;

    Gx = input[ww*y + x+1] - input[ww*y + x-1];
    Gy = input[ww*(y+1) + x] - input[ww*(y-1) + x];

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void matrix_normalization(void) {
    memset(normalization, 0, ww*hh*sizeof (double));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < ww - 1; x += 1) {
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    normalization[ww*y + x] += weights[ww*(y+i) + x+j];
                }
            }
        }
    }
}

void matrix_convolute(void) {
    memset(output, 0, ww*hh*sizeof (double));
    for (int32 y = 1; y < hh - 1; y += 1) {
        for (int32 x = 1; x < ww - 1; x += 1) {
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    output[ww*y + x] += (weights[ww*(y+i) + x+j]*input[ww*(y+i) + x+j]);
                }
            }
            output[ww*y + x] /= normalization[ww*y + x];
        }
    }
    for (int32 x = 0; x < (ww*hh - 1); x += ww)
        output[x] = output[x+1];
    for (int32 y = 0; y < ww-1; y += 1)
        output[y] = output[y+ww];
    for (int32 x = ww-1; x < (ww*hh - 1); x += ww)
        output[x] = output[x-1];
    for (int32 y = (hh-1)*ww; y < (ww*hh - 1); y += 1)
        output[y] = output[y-ww];
}
