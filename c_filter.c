/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <unistd.h>
#include <math.h>
typedef int32_t int32;

static int32 xx;
static int32 yy;

static inline double weight(double * restrict, int32, int32);
static void matrix_weights(double * restrict, double * restrict);
static void matrix_normalization(double * restrict, double * restrict);
static void matrix_convolute(double * restrict, double * restrict,
                             double * restrict, double * restrict);

void filter(double * restrict input, int32 const ww, int32 const hh, 
            double * restrict weights, double * restrict normalization, 
            double * restrict output) {
    xx = ww;
    yy = hh;
    matrix_weights(input, weights);
    matrix_normalization(weights, normalization);
    matrix_convolute(input, weights, normalization, output);
}

double weight(double * restrict input, int32 x, int32 y) {
    double Gx, Gy;
    double d, w;

    Gx = input[yy*(x+1) + y] - input[yy*(x-1) + y];
    Gy = input[yy*x + y+1] - input[yy*x + y-1];

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

typedef struct ThreadArguments {
    double *input;
    double *weights;
    int32 start_x;
    int32 end_x;
} ThreadArguments;

int weights_slice(void *arg) {
    ThreadArguments *args = (ThreadArguments *) arg;

    double *input = args->input;
    double *weights = args->weights;
    int32 start_x = args->start_x;
    int32 end_x = args->end_x;

    for (int32 x = start_x; x < end_x; x++) {
        for (int32 y = 1; y < yy - 1; y++) {
            weights[yy*x + y] = weight(input, x, y);
        }
    }

    thrd_exit(0);
}

void matrix_weights(double *restrict input, double *restrict weights) {
    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
    int32 range = (xx - 2) / number_threads;
    
    thrd_t threads[number_threads];
    ThreadArguments thread_arguments[number_threads];

    for (int i = 0; i < number_threads; i++) {
        thread_arguments[i].input = input;
        thread_arguments[i].weights = weights;
        thread_arguments[i].start_x = i*range + 1;
        if (i == number_threads - 1) {
            thread_arguments[i].end_x = xx - 1;
        } else {
            thread_arguments[i].end_x = (i + 1)*range + 1;
        }

        thrd_create(&threads[i], weights_slice, (void *) &thread_arguments[i]);
    }

    for (int i = 0; i < number_threads; i++) {
        thrd_join(threads[i], NULL);
    }
}

void matrix_normalization(double * restrict weights, double * restrict normalization) {
    for (int32 x = 1; x < xx - 1; x++) {
        for (int32 y = 1; y < yy - 1; y++) {
            normalization[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i++) {
                for (int32 j = -1; j <= +1; j++) {
                    normalization[yy*x + y] += weights[yy*(x+i) + y+j];
                }
            }
        }
    }
}

void matrix_convolute(double * restrict input, double * restrict weights, 
                      double * restrict normalization, double * restrict output) {
    for (int32 x = 1; x < xx - 1; x++) {
        for (int32 y = 1; y < yy - 1; y++) {
            output[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i++) {
                for (int32 j = -1; j <= +1; j++) {
                    output[yy*x + y] += (weights[yy*(x+i) + y+j]*input[yy*(x+i) + y+j]);
                }
            }
            output[yy*x + y] /= normalization[yy*x + y];
        }
    }
    for (int32 y = 0; y < (yy*xx - 1); y+=yy)
        output[y] = output[y+1];
    for (int32 x = 0; x < yy-1; x++)
        output[x] = output[x+yy];
    for (int32 y = yy-1; y < (yy*xx - 1); y+=yy)
        output[y] = output[y-1];
    for (int32 x = (xx-1)*yy; x < (yy*xx - 1); x++)
        output[x] = output[x-yy];
}
