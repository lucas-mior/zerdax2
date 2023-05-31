/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
typedef int32_t int32;

static int32 xx;
static int32 yy;

static inline double weight(double * restrict, int32, int32);
static void matrix_weight(double * restrict, double * restrict);
static void matrix_normalize(double * restrict, double * restrict);
static void matrix_convolute(double * restrict, double * restrict,
                             double * restrict, double * restrict);

void filter(double * restrict input, int32 const ww, int32 const hh, 
            double * restrict W, double * restrict N, 
            double * restrict output) {
    xx = ww;
    yy = hh;
    matrix_weight(input, W);
    matrix_normalize(W, N);
    matrix_convolute(input, W, N, output);
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
    double *W;
    int32_t start_y;
    int32_t end_y;
} ThreadArguments;

void *weights_row(void *arg) {
    ThreadArguments *args = (ThreadArguments *) arg;

    double *input = args->input;
    double *W = args->W;
    int32_t start_y = args->start_y;
    int32_t end_y = args->end_y;

    for (int32_t y = start_y; y < end_y; y++) {
        for (int32_t x = 1; x < xx - 1; x++) {
            W[yy*x + y] = weight(input, x, y);
        }
    }

    pthread_exit(NULL);
}

void matrix_weight(double *restrict input, double *restrict W) {
    long number_threads = sysconf(_SC_NPROCESSORS_ONLN);
    int32_t range = (yy - 2) / number_threads;
    
    pthread_t threads[number_threads];
    ThreadArguments thread_arguments[number_threads];

    for (int i = 0; i < number_threads; i++) {
        thread_arguments[i].input = input;
        thread_arguments[i].W = W;
        thread_arguments[i].start_y = i*range + 1;
        if (i == number_threads - 1) {
            thread_arguments[i].end_y = yy - 1;
        } else {
            thread_arguments[i].end_y = (i + 1)*range + 1;
        }

        pthread_create(&threads[i], NULL, weights_row, (void *) &thread_arguments[i]);
    }

    for (int i = 0; i < number_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void matrix_normalize(double * restrict W, double * restrict N) {
    for (int32 x = 1; x < xx - 1; x++) {
        for (int32 y = 1; y < yy - 1; y++) {
            N[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i++) {
                for (int32 j = -1; j <= +1; j++) {
                    N[yy*x + y] += W[yy*(x+i) + y+j];
                }
            }
        }
    }
}

void matrix_convolute(double * restrict input, double * restrict W, 
                      double * restrict N, double * restrict output) {
    for (int32 x = 1; x < xx - 1; x++) {
        for (int32 y = 1; y < yy - 1; y++) {
            output[yy*x + y] = 0;
            for (int32 i = -1; i <= +1; i++) {
                for (int32 j = -1; j <= +1; j++) {
                    output[yy*x + y] += (W[yy*(x+i) + y+j]*input[yy*(x+i) + y+j]);
                }
            }
            output[yy*x + y] /= N[yy*x + y];
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
