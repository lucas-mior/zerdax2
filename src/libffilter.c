#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double weight(double *f, int32_t x, int32_t y, int32_t yy, double h) {
    double Gx, Gy;
    double d, w;

    Gx = (f[yy*(x+1) + y] - f[yy*(x-1) + y]) / 2;
    Gy = (f[yy*x + y+1] - f[yy*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d)/h);
    return w;
}

void weight_array(double *f, int32_t xx, int32_t yy, double *W, double h) {
    for (int32_t x = 1; x < xx-1; x++) {
        for (int32_t y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(f, x, y, yy, h);
        }
    }
}

void norm_array(int32_t xx, int32_t yy, double *W, double *N) {
    for (int32_t x = 1; x < xx - 1; x++) {
        for (int32_t y = 1; y < yy - 1; y++) {
            N[yy*x + y] = 0;
            for (int32_t i = -1; i <= +1; i++) {
                for (int32_t j = -1; j <= +1; j++) {
                    N[yy*x + y] += W[yy*(x+i) + y+j];
                }
            }
        }
    }
}

void convolute(double *f, int32_t xx, int32_t yy, double *W, double *N, double *g) {
    for (int32_t x = 1; x < xx - 1; x++) {
        for (int32_t y = 1; y < yy - 1; y++) {
            g[yy*x + y] = 0;
            for (int32_t i = -1; i <= +1; i++) {
                for (int32_t j = -1; j <= +1; j++) {
                    g[yy*x + y] += (W[yy*(x+i) + y+j]*f[yy*(x+i) + y+j])/N[yy*x + y];
                }
            }
        }
    }
    for (int32_t y = 0; y < (yy*xx - 1); y+=yy)
        g[y] = g[y+1];
    for (int32_t x = 0; x < yy-1; x++)
        g[x] = g[x+yy];
    for (int32_t y = yy-1; y < (yy*xx - 1); y+=yy)
        g[y] = g[y-1];
    for (int32_t x = (xx-1)*yy; x < (yy*xx - 1); x++)
        g[x] = g[x-yy];
}

void ffilter(double *f, int32_t xx, int32_t yy, double *W, double *N, double *g, double h) {
    weight_array(f, xx, yy, W, h);
    norm_array(xx, yy, W, N);
    convolute(f, xx, yy, W, N, g);
}
