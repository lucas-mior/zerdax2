#ifndef C_FILTER_C
#define C_FILTER_C

/* Filter proposed by Bing Wang and ShaoSheng Fan
 * "An improved CANNY edge detection algorithm"
 * 2009 Second International Workshop on Computer Science and Engineering */

#if defined(__INCLUDE_LEVEL__) && (__INCLUDE_LEVEL__ == 0)
#define TESTING_c_filter 1
#elif !defined(TESTING_c_filter)
#define TESTING_c_filter 0
#endif

#define CBASE_IMPLEMENT
#include "cbase.h"

#define WW0 512
#define MAX_THREADS 8

#define USE_DOUBLE 1

#if USE_DOUBLE
typedef double floaty;
#else
typedef float floaty;
#endif

static const int32 WW = WW0;

static floaty *restrict input;
static floaty *restrict weights;
static floaty *restrict output;
static int32 hh;
static int32 nthreads;
static int32 matrix_size;

void filter(floaty *restrict, floaty *restrict,
            floaty *restrict, int32, int32);

typedef struct Slice {
    int32 y0;
    int32 y1;
    int32 id;
} Slice;

static pthread_mutex_t mutexes[MAX_THREADS];
static pthread_mutex_t all_locks_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t all_locks_cond = PTHREAD_COND_INITIALIZER;
static int32 all_locks_count;

static void
xcond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    int err;

    if ((err = pthread_cond_wait(cond, mutex))) {
        error("Error waiting for cond %p: %s.\n", (void *)cond,
              strerror(err));
        fatal(EXIT_FAILURE);
    }
    return;
}

static void
xcond_broadcast(pthread_cond_t *cond) {
    int err;

    if ((err = pthread_cond_broadcast(cond))) {
        error("Error broadcasting cond %p: %s.\n", (void *)cond,
              strerror(err));
        fatal(EXIT_FAILURE);
    }
    return;
}

static void
wait_for_all_slice_locks(void) {
    xpthread_mutex_lock(&all_locks_mutex);
    all_locks_count += 1;
    if (all_locks_count == nthreads) {
        xcond_broadcast(&all_locks_cond);
    }
    while (all_locks_count < nthreads) {
        xcond_wait(&all_locks_cond, &all_locks_mutex);
    }
    xpthread_mutex_unlock(&all_locks_mutex);
    return;
}

static void
wait_for_slice_weights(int32 id) {
    xpthread_mutex_lock(&mutexes[id]);
    xpthread_mutex_unlock(&mutexes[id]);
    return;
}

static void *
work(void *arg) {
    Slice *slice = arg;
    int32 y0 = slice->y0;
    int32 y1 = slice->y1;
    int32 id = slice->id;

    int32 clear_y0 = y0;
    int32 clear_dy;

    if (id > 0) {
        clear_y0 += 1;
    }
    clear_dy = y1 - clear_y0 + 1;
    if (y1 == (hh - 2)) {
        clear_dy += 1;
    }

    xpthread_mutex_lock(&mutexes[id]);
    wait_for_all_slice_locks();

    memset64(&(output[clear_y0*WW]), 0, clear_dy*WW*SIZEOF(*output));
    memset64(&(weights[clear_y0*WW]), 0, clear_dy*WW*SIZEOF(*weights));

    for (int32 y = y0 + 1; y < (y1 + 1); y += 1) {
        for (int32 x = 1; x < (WW - 1); x += 1) {
            floaty Gx, Gy;
            floaty d, w;
            floaty xx;

            Gx = input[WW*y + x+1] - input[WW*y + x-1];
            Gy = input[WW*(y+1) + x] - input[WW*(y-1) + x];

            /* xx = fma(Gx, Gx, Gy*Gy); */
            xx = Gx*Gx + Gy*Gy;
            d = sqrt(xx);
            w = exp(-sqrt(d));
            weights[WW*y + x] = w;
        }
    }

    xpthread_mutex_unlock(&mutexes[id]);

    if (id > 0) {
        wait_for_slice_weights(id - 1);
    }
    if (id < (nthreads - 1)) {
        wait_for_slice_weights(id + 1);
    }

    for (int32 y = y0 + 1; y < (y1 + 1); y += 1) {
        for (int32 x = 1; x < (WW - 1); x += 1) {
            floaty norm = 0;
            for (int32 i = -1; i <= +1; i += 1) {
                for (int32 j = -1; j <= +1; j += 1) {
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
       floaty *restrict weights0, int32 hh0, int32 nthreads0) {
    pthread_t threads[MAX_THREADS];
    Slice slices[MAX_THREADS];
    int32 range;

    input = input0;
    weights = weights0;
    output = output0;
    hh = hh0;
    matrix_size = WW * hh;

    if (nthreads0 < 1) {
        nthreads = 1;
    } else if (nthreads0 > MAX_THREADS) {
        nthreads = MAX_THREADS;
    } else {
        nthreads = nthreads0;
    }

    range = hh / nthreads;

    all_locks_count = 0;
    for (int32 i = 0; i < nthreads; i += 1) {
        xpthread_mutex_init(&mutexes[i], NULL);
    }

    for (int32 i = 0; i < (nthreads - 1); i += 1) {
        slices[i].y0 = i*range;
        slices[i].y1 = (i + 1)*range;
        slices[i].id = i;

        xpthread_create(&threads[i], NULL, work, (void *)&slices[i]);
    }{
        int32 i = nthreads - 1;
        slices[i].y0 = i*range;
        slices[i].y1 = hh - 2;
        slices[i].id = i;

        xpthread_create(&threads[i], NULL, work, (void *)&slices[i]);
    }

    for (int32 i = 0; i < nthreads; i += 1) {
        xpthread_join(&threads[i], NULL);
    }
    for (int32 i = 0; i < nthreads; i += 1) {
        xpthread_mutex_destroy(&mutexes[i]);
    }

    for (int32 x = 0; x < (matrix_size - 1); x += WW) {
        output[x] = output[x+1];
    }
    for (int32 y = 0; y < (WW - 1); y += 1) {
        output[y] = output[y+WW];
    }
    for (int32 x = WW - 1; x < (matrix_size - 1); x += WW) {
        output[x] = output[x-1];
    }
    for (int32 y = (hh - 1)*WW; y < (matrix_size - 1); y += 1) {
        output[y] = output[y-WW];
    }

    return;
}

#if TESTING_c_filter
#define HH0 512
#define IMAGE_SIZE HH0*WW0

static uint64
hash_array(floaty *array) {
    uint64 hash = 5381;
    for (int32 i = 0; i < IMAGE_SIZE; i += 1) {
        uint64 c = 0;
        memcpy64(&c, &array[i], SIZEOF(*array));
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static uint8
image_sample(floaty value) {
    ASSERT(isfinite(value));

    if (value <= 0) {
        return 0;
    }
    if (value >= 255) {
        return 255;
    }

    return (uint8)round(value);
}

static uint64
hash_image_array(floaty *array) {
    uint64 hash = 5381;
    for (int32 i = 0; i < IMAGE_SIZE; i += 1) {
        uint64 c = image_sample(array[i]);
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static inline floaty
randd(void) {
    int64 r;

    while ((r = rand_int()) < 0) {
    }
    return (floaty)(r % 256);
}

typedef struct SaveHash {
    uint32 w;
    uint32 h;
    uint32 use_double;
    uint32 unused;
    uint64 hash_input;
    uint64 hash_output;
} SaveHash;

#define LENGHT(X) (int32)(sizeof(X) / sizeof(*X))
static SaveHash hash_remember[] = {
    {512, 512, 1, 0, 154476712296453381ull, 9060896705864926636ull},
};

int32 main(int32 argc, char **argv) {
    int32 hh0 = HH0;
    int32 nfilters = 500;
    bool save_results = false;
    uint64 hash_input;
    uint64 hash_output;

    floaty *input0 = malloc2(IMAGE_SIZE*sizeof(*input0));
    floaty *output0 = malloc2(IMAGE_SIZE*sizeof(*output0));
    floaty *weights0 = malloc2(IMAGE_SIZE*sizeof(*weights0));

    struct timespec t0, t1;
    (void) argc;
    (void) argv;

    save_results = argc > 1;

    for (int32 i = 0; i < IMAGE_SIZE; i += 4) {
        input0[i+0] = randd();
        input0[i+1] = randd();
        input0[i+2] = randd();
        input0[i+3] = randd();
    }

    hash_input = hash_array(input0);
    printf("input hash: %lluull\n", hash_input);
    clock_gettime(CLOCK_REALTIME, &t0);

    nthreads = (int32) sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads < 1)
        nthreads = 1;
    else if (nthreads > MAX_THREADS)
        nthreads = MAX_THREADS;
    
    for (int32 i = 0; i < nfilters; i += 1) {
        filter(input0, output0, weights0, hh0, nthreads);
    }

    clock_gettime(CLOCK_REALTIME, &t1);

    hash_output = hash_image_array(output0);
    printf("output hash: %lluull\n", hash_output);

    for (int32 i = 0; i < LENGHT(hash_remember); i += 1) {
        SaveHash save_hash = hash_remember[i];
        if ((save_hash.w == WW0) 
            && (save_hash.h == HH0)
            && (save_hash.use_double == USE_DOUBLE)) {
             ASSERT_EQUAL(hash_output, save_hash.hash_output);
             ASSERT_EQUAL(hash_input, save_hash.hash_input);
             break;
        }
    }

    {
        int64 seconds = t1.tv_sec - t0.tv_sec;
        int64 nanos = t1.tv_nsec - t0.tv_nsec;

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
        FILE *image1;
        FILE *image2;
        int64 written;

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

        written = fwrite64(input0, sizeof(*input0), IMAGE_SIZE, image1);
        if (written < IMAGE_SIZE) {
            error("Error writing to %s: %s.\n", input_file, strerror(errno));
        }

        written = fwrite64(output0, sizeof(*output0), IMAGE_SIZE, image2);
        if (written < IMAGE_SIZE) {
            error("Error writing to %s: %s.\n", output_file, strerror(errno));
        }

        if (fclose(image1)) {
            error("Error closing %s: %s.\n", input_file, strerror(errno));
        }
        if (fclose(image2)) {
            error("Error closing %s: %s.\n", output_file, strerror(errno));
        }
    }
    free2(input0, IMAGE_SIZE*sizeof(*input0));
    free2(output0, IMAGE_SIZE*sizeof(*output0));
    free2(weights0, IMAGE_SIZE*sizeof(*weights0));

    return 0;
}
#endif /* TESTING_c_filter */

#endif /* C_FILTER_C */
