/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <stdint.h>
#include <math.h>
#include <immintrin.h> // Include the header for SIMD intrinsics
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

// Define constants for the polynomial approximation
const double EXP_POLY_C1 = 1.0;
const double EXP_POLY_C2 = 0.041666666666666664;
const double EXP_POLY_C3 = 0.0013888888888888888;
const double EXP_POLY_C4 = 2.48015873015873e-05;
const double EXP_POLY_C5 = 2.7557319223985893e-07;
const double EXP_POLY_C6 = 2.5052108385441714e-09;

void matrix_weight(double * restrict input, double * restrict W) {
    const int32 simdWidth = 4;
    const int32 unrollSize = (yy - 1) / simdWidth * simdWidth;

    // Load the polynomial coefficients into SIMD registers
    __m256d c1Vec = _mm256_set1_pd(EXP_POLY_C1);
    __m256d c2Vec = _mm256_set1_pd(EXP_POLY_C2);
    __m256d c3Vec = _mm256_set1_pd(EXP_POLY_C3);
    __m256d c4Vec = _mm256_set1_pd(EXP_POLY_C4);
    __m256d c5Vec = _mm256_set1_pd(EXP_POLY_C5);
    __m256d c6Vec = _mm256_set1_pd(EXP_POLY_C6);

    for (int32 x = 1; x < xx - 1; x++) {
        for (int32 y = 1; y < unrollSize; y += simdWidth) {
            __m256d GxVec = _mm256_sub_pd(_mm256_loadu_pd(&input[yy * (x + 1) + y]), _mm256_loadu_pd(&input[yy * (x - 1) + y]));
            __m256d GyVec = _mm256_sub_pd(_mm256_loadu_pd(&input[yy * x + y + 1]), _mm256_loadu_pd(&input[yy * x + y - 1]));

            __m256d dVec = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(GxVec, GxVec), _mm256_mul_pd(GyVec, GyVec)));

            // Approximate the exponential function using a polynomial approximation
            __m256d term1 = _mm256_mul_pd(dVec, c6Vec);
            __m256d term2 = _mm256_add_pd(_mm256_mul_pd(term1, dVec), c5Vec);
            __m256d term3 = _mm256_add_pd(_mm256_mul_pd(term2, dVec), c4Vec);
            __m256d term4 = _mm256_add_pd(_mm256_mul_pd(term3, dVec), c3Vec);
            __m256d term5 = _mm256_add_pd(_mm256_mul_pd(term4, dVec), c2Vec);
            __m256d term6 = _mm256_add_pd(_mm256_mul_pd(term5, dVec), c1Vec);
            __m256d expVec = _mm256_add_pd(term6, dVec);

            _mm256_storeu_pd(&W[yy * x + y], expVec);
        }

        for (int32 y = unrollSize; y < yy - 1; y++) {
            W[yy * x + y] = weight(input, x, y);
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
