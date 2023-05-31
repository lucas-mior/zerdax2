/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
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

#define NUM_THREADS 4

typedef struct {
    double *input;
    double *W;
    int32_t xx;
    int32_t yy;
    int32_t start_x;
    int32_t end_x;
} ThreadArgs;

void *computeWeights(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    double *input = args->input;
    double *W = args->W;
    /* int32_t xx = args->xx; */
    int32_t yy = args->yy;
    int32_t start_x = args->start_x;
    int32_t end_x = args->end_x;

    for (int32_t x = start_x; x < end_x; x++) {
        for (int32_t y = 1; y < yy - 1; y++) {
            W[yy * x + y] = weight(input, x, y);
        }
    }

    pthread_exit(NULL);
}

void matrix_weight(double *restrict input, double *restrict W) {
    int32_t range = (xx - 2) / NUM_THREADS;
    
    pthread_t threads[NUM_THREADS];
    ThreadArgs threadArgs[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        threadArgs[i].input = input;
        threadArgs[i].W = W;
        threadArgs[i].xx = xx;
        threadArgs[i].yy = yy;
        threadArgs[i].start_x = i * range + 1;
        threadArgs[i].end_x = (i == NUM_THREADS - 1) ? xx - 1 : (i + 1) * range + 1;

        pthread_create(&threads[i], NULL, computeWeights, (void *)&threadArgs[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
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
