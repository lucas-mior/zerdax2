#ifndef C_FILTER_C
#define C_FILTER_C

/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#pragma push_macro("TESTING_THIS_FILE")
#define TESTING_THIS_FILE 0

#include "c_util.c"
#include "c_segments.c"
#include "c_lines_bundle.c"

#pragma pop_macro("TESTING_THIS_FILE")

#define WW0 512
#define MAX_THREADS 8

#define USE_DOUBLE 0

#if USE_DOUBLE
typedef double floaty;
#define SQRT sqrt
#define EXP exp
#else
typedef float floaty;
#define SQRT sqrtf
#define EXP expf
#endif

static const int WW = WW0;

static floaty *restrict input;
static floaty *restrict weights;
static floaty *restrict output;
static int hh;
static int nthreads;
static int matrix_size;

void filter(floaty *restrict, floaty *restrict,
            floaty *restrict, int, int);

typedef struct Slice {
    int y0;
    int y1;
    int id;
} Slice;

static pthread_mutex_t mutexes[MAX_THREADS];

static void *
work(void *arg) {
    Slice *slice = arg;
    int y0 = slice->y0;
    int y1 = slice->y1;
    int id = slice->id;

    int dy = y1 - y0 + 1;
    if (y1 == (hh - 2))
        dy += 1;

    memset(&(output[y0*WW]), 0, (size_t)dy*WW*sizeof(*output));
    memset(&(weights[y0*WW]), 0, (size_t)dy*WW*sizeof(*weights));

    for (int y = y0 + 1; y < (y1 + 1); y += 1) {
        for (int x = 1; x < (WW - 1); x += 1) {
            floaty Gx, Gy;
            floaty d, w;

            Gx = input[WW*y + x+1] - input[WW*y + x-1];
            Gy = input[WW*(y+1) + x] - input[WW*(y-1) + x];

            d = SQRT(Gx*Gx + Gy*Gy);
            w = EXP(-SQRT(d));
            weights[WW*y + x] = w;
        }
    }

    pthread_mutex_unlock(&mutexes[id]);

    if (id < (nthreads - 1)) {
        pthread_mutex_lock(&mutexes[id + 1]);
        pthread_mutex_unlock(&mutexes[id + 1]);
    }
    if (id > 0) {
        pthread_mutex_lock(&mutexes[id - 1]);
        pthread_mutex_unlock(&mutexes[id - 1]);
    }

    for (int y = y0 + 1; y < (y1 + 1); y += 1) {
        for (int x = 1; x < (WW - 1); x += 1) {
            floaty norm = 0;
            for (int i = -1; i <= +1; i += 1) {
                for (int j = -1; j <= +1; j += 1) {
                    floaty w = weights[WW*(y+i) + x+j];
                    norm += w;
                    output[WW*y + x] += w*input[WW*(y+i) + x+j];
                }
            }
            output[WW*y + x] /= norm;
        }
    }

    pthread_exit(0);
}

void
filter(floaty *restrict input0, floaty *restrict output0,
       floaty *restrict weights0, int hh0, int nthreads0) {
    pthread_t threads[MAX_THREADS];
    Slice slices[MAX_THREADS];
    int range;

    input = input0;
    weights = weights0;
    output = output0;
    hh = hh0;
    matrix_size = WW * hh;

    if (nthreads0 < 1)
        nthreads = 1;
    else if (nthreads0 > MAX_THREADS)
        nthreads = MAX_THREADS;
    else
        nthreads = nthreads0;

    range = hh / nthreads;

    for (int i = 0; i < nthreads; i += 1) {
        pthread_mutex_init(&mutexes[i], NULL);
        pthread_mutex_lock(&mutexes[i]);
    }

    for (int i = 0; i < (nthreads - 1); i += 1) {
        slices[i].y0 = i*range;
        slices[i].y1 = (i + 1)*range;
        slices[i].id = i;

        pthread_create(&threads[i], NULL, work, (void *)&slices[i]);
    }{
        int i = nthreads - 1;
        slices[i].y0 = i*range;
        slices[i].y1 = hh - 2;
        slices[i].id = i;

        pthread_create(&threads[i], NULL, work, (void *)&slices[i]);
    }

    for (int i = 0; i < nthreads; i += 1)
        pthread_join(threads[i], NULL);

    for (int x = 0; x < (matrix_size - 1); x += WW)
        output[x] = output[x+1];
    for (int y = 0; y < (WW - 1); y += 1)
        output[y] = output[y+WW];
    for (int x = WW - 1; x < (matrix_size - 1); x += WW)
        output[x] = output[x-1];
    for (int y = (hh - 1)*WW; y < (matrix_size - 1); y += 1)
        output[y] = output[y-WW];

    return;
}

#ifndef TESTING_THIS_FILE
#define TESTING_THIS_FILE 0
#endif

#if TESTING_THIS_FILE
#include <errno.h>
#define HH0 512
#define IMAGE_SIZE HH0*WW0

static unsigned long
hash_function(floaty *array) {
    unsigned long hash = 5381;
    for (int i = 0; i < IMAGE_SIZE; i += 1) {
        unsigned long c = 0;
        memcpy(&c, &array[i], sizeof(*array));
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static inline floaty
randd(void) {
    long r;
    while ((r = rand()) < 0);
    return (floaty)(double)r;
}

typedef uint32_t uint32;
typedef uint64_t uint64;

typedef struct SaveHash {
    uint32 w;
    uint32 h;
    uint32 use_double;
    uint32 unused;
    uint64 hash_input;
    uint64 hash_output;
} SaveHash;

#define LENGHT(X) (int)(sizeof(X) / sizeof(*X))
static SaveHash hash_remember[] = {
    {512, 512, 0, 0, 8707747967837504398u, 6217956780236870917u},
    {1080, 1080, 0, 0, 13196852808646899663u, 11178258618305559813u},
};

int main(int argc, char **argv) {
    int hh0 = HH0;
    int nfilters = 2000;
    bool save_results = false;
    uint64 hash_input;
    uint64 hash_output;

    floaty *input0 = util_malloc(IMAGE_SIZE*sizeof(*input0));
    floaty *output0 = util_malloc(IMAGE_SIZE*sizeof(*output0));
    floaty *weights0 = util_malloc(IMAGE_SIZE*sizeof(*weights0));

    struct timespec t0, t1;
    (void) argc;
    (void) argv;

    save_results = argc > 1;

    for (int i = 0; i < IMAGE_SIZE; i += 4) {
        input0[i+0] = randd();
        input0[i+1] = randd();
        input0[i+2] = randd();
        input0[i+3] = randd();
    }

    hash_input = hash_function(input0);
    printf("input hash: %luu\n", hash_input);
    clock_gettime(CLOCK_REALTIME, &t0);

    nthreads = (int) sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1)
        nthreads = 1;
    else if (nthreads > MAX_THREADS)
        nthreads = MAX_THREADS;
    
    for (int i = 0; i < nfilters; i += 1)
        filter(input0, output0, weights0, hh0, nthreads);

    clock_gettime(CLOCK_REALTIME, &t1);

    hash_output = hash_function(output0);
    printf("output hash: %luu\n", hash_output);

    for (int i = 0; i < LENGHT(hash_remember); i += 1) {
        SaveHash save_hash = hash_remember[i];
        if ((save_hash.w == WW0) && (save_hash.h == HH0)
            && (save_hash.use_double == USE_DOUBLE)) {
             assert(hash_output == save_hash.hash_output);
             assert(hash_input == save_hash.hash_input);
             break;
        }
    }

    {
        long seconds = t1.tv_sec - t0.tv_sec;
        long nanos = t1.tv_nsec - t0.tv_nsec;

        double total_seconds = (double)seconds + (double)nanos/1.0e9;
        double micros_per_filter = 1e6*(total_seconds/(double)nfilters);
        double nanos_per_pixel = 1e3*(micros_per_filter/((double)IMAGE_SIZE));
        double fps = (double)nfilters/total_seconds;

        printf("%s:\n", __FILE__);
        printf("%gs = %gus per filter = %gns per pixel = %uHz\n",
               total_seconds, micros_per_filter, nanos_per_pixel, (uint)fps);
    }

    if (save_results) {
        char *input_file = "input.data";
        char *output_file = "output.data";
        size_t written;

        FILE *image1;
        FILE *image2;

        if ((image1 = fopen(input_file, "w")) == NULL) {
            error("Error opening '%s' for writing: %s.\n",
                  input_file, strerror(errno));
            exit(EXIT_FAILURE);
        }
        if ((image2 = fopen(output_file, "w")) == NULL) {
            error("Error opening '%s' for writing: %s.\n",
                  output_file, strerror(errno));
            exit(EXIT_FAILURE);
        }

        written = fwrite(input0, sizeof(*input0), IMAGE_SIZE, image1);
        if (written < IMAGE_SIZE)
            error("Error writing to %s: %s.\n", input_file, strerror(errno));

        written = fwrite(output0, sizeof(*output0), IMAGE_SIZE, image2);
        if (written < IMAGE_SIZE)
            error("Error writing to %s: %s.\n", output_file, strerror(errno));

        if (fclose(image1))
            error("Error closing %s: %s.\n", input_file, strerror(errno));
        if (fclose(image2))
            error("Error closing %s: %s.\n", output_file, strerror(errno));
    }
    free(input0);
    free(output0);
    free(weights0);

    return 0;
}
#endif

#endif
