/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
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

    Gx = (input[yy*(x+1) + y] - input[yy*(x-1) + y]) / 2;
    Gy = (input[yy*x + y+1] - input[yy*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void matrix_weight(double * restrict input, double * restrict W) {
    for (int32 x = 1; x < xx-1; x++) {
        for (int32 y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(input, x, y);
        }
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
