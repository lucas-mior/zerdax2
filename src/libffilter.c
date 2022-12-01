#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double weight(double *f, int x, int y, int yy) {
    double Gx, Gy;
    double d, w;

    Gx = (f[yy*(x+1) + y] - f[yy*(x-1) + y]) / 2;
    Gy = (f[yy*x + y+1] - f[yy*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d)/1);
    return w;
}

void weight_array(double *f, int xx, int yy, double *W) {
    for (int x = 1; x < xx-1; x++) {
        for (int y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(f, x, y, yy);
        }
    }
}

void norm_array(int xx, int yy, double *W, double *N) {
    for (int x = 1; x < xx - 1; x++) {
        for (int y = 1; y < yy - 1; y++) {
            N[yy*x + y] = 0;
            for (int i = -1; i <= +1; i++) {
                for (int j = -1; j <= +1; j++) {
                    N[yy*x + y] += W[yy*(x+i) + y+j];
                }
            }
        }
    }
}

void convolute(double *f, int xx, int yy, double *W, double *N, double *g) {
    for (int x = 1; x < xx - 1; x++) {
        for (int y = 1; y < yy - 1; y++) {
            g[yy*x + y] = 0;
            for (int i = -1; i <= +1; i++) {
                for (int j = -1; j <= +1; j++) {
                    g[yy*x + y] += (W[yy*(x+i) + y+j]*f[yy*(x+i) + y+j])/N[yy*x + y];
                }
            }
        }
    }
}

void ffilter(double *f, int xx, int yy, double *W, double *N, double *g) {
    weight_array(f, xx, yy, W);
    norm_array(xx, yy, W, N);
    convolute(f, xx, yy, W, N, g);
}
