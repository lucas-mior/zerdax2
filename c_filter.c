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

static double *restrict input;
static double *restrict weights;
static double *restrict normalization;
static double *restrict output;
static int32 xx;
static int32 yy;

static inline double weight(int32, int32);
static void matrix_weights(void);
static void matrix_normalization(void);
static void matrix_convolute(void);
static int weights_slice(void *);

void filter(double *restrict input0, double *restrict output0, 
            double *restrict normalization0, double *restrict weights0,
            int32 const xx0, int32 const yy0) {

    input = input0;
    weights = weights0;
    normalization = normalization0;
    output = output0;
    xx = xx0;
    yy = yy0;

    matrix_weights();
    matrix_normalization();
    matrix_convolute();
}

typedef struct ThreadArguments {
    int32 start_x;
    int32 end_x;
} ThreadArguments;

void matrix_weights(void) {
    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (number_threads > 4)
        number_threads = 4;
    int32 range = (xx - 2) / number_threads;
    
    thrd_t threads[number_threads];
    ThreadArguments thread_arguments[number_threads];

    for (int i = 0; i < number_threads; i += 1) {
        thread_arguments[i].start_x = i*range + 1;
        if (i == number_threads - 1) {
            thread_arguments[i].end_x = xx - 1;
        } else {
            thread_arguments[i].end_x = (i + 1)*range + 1;
        }

        thrd_create(&threads[i], weights_slice, (void *) &thread_arguments[i]);
    }

    for (int i = 0; i < number_threads; i += 1) {
        thrd_join(threads[i], NULL);
    }
}

int weights_slice(void *arg) {
    ThreadArguments *args = (ThreadArguments *) arg;

    int32 start_x = args->start_x;
    int32 end_x = args->end_x;

    for (int32 x = start_x; x < end_x; x += 1) {
        for (int32 y = 1; y < yy - 1; y += 1) {
            weights[yy*x + y] = weight(x, y);
        }
    }

    thrd_exit(0);
}

double weight(int32 x, int32 y) {
    double Gx, Gy;
    double d, w;

    Gx = input[yy*(x+1) + y] - input[yy*(x-1) + y];
    Gy = input[yy*x + y+1] - input[yy*x + y-1];

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void matrix_normalization(void) {
    for (int32 x = 1; x < xx - 1; x += 1) {
        for (int32 y = 1; y < yy - 1; y += 1) {
            normalization[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    normalization[yy*x + y] += weights[yy*(x+i) + y+j];
                }
            }
        }
    }
}

void matrix_convolute(void) {
    for (int32 x = 1; x < xx - 1; x += 1) {
        for (int32 y = 1; y < yy - 1; y += 1) {
            output[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
                    output[yy*x + y] += (weights[yy*(x+i) + y+j]*input[yy*(x+i) + y+j]);
                }
            }
            output[yy*x + y] /= normalization[yy*x + y];
        }
    }
    for (int32 y = 0; y < (yy*xx - 1); y += yy)
        output[y] = output[y+1];
    for (int32 x = 0; x < yy-1; x += 1)
        output[x] = output[x+yy];
    for (int32 y = yy-1; y < (yy*xx - 1); y += yy)
        output[y] = output[y-1];
    for (int32 x = (xx-1)*yy; x < (yy*xx - 1); x += 1)
        output[x] = output[x-yy];
}
