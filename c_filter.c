/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <math.h>
typedef int32_t int32;

static int32 xx;
static int32 yy;
static double h;

static inline double weight(double * restrict input, int32 x, int32 y) {
    double Gx, Gy;
    double d, w;

    Gx = (input[yy*(x+1) + y] - input[yy*(x-1) + y]) / 2;
    Gy = (input[yy*x + y+1] - input[yy*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d)/h);
    return w;
}

static void weight_array(double * restrict input, double * restrict W) {
    for (int32 x = 1; x < xx-1; x++) {
        for (int32 y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(input, x, y);
        }
    }
}

static void norm_array(double * restrict W, double * restrict N) {
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

static void convolute(double * restrict input, double * restrict W, 
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

void filter(double * restrict input, int32 const ww, int32 const hh, 
            double * restrict W, double * restrict N, 
            double * restrict output, double const h0) {
    xx = ww;
    yy = hh;
    h = h0;
    weight_array(input, W);
    norm_array(W, N);
    convolute(input, W, N, output);
}
