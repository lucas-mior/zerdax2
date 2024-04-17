#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "c_declarations.h"

static inline int32 minimum(int32 const [4]);
static bool segments_intersect(int32 *restrict, int32 *restrict);
static int32 distance_point_segment(int32 const, int32 const, int32 *restrict);

int32
segments_distance(int32 *restrict line0, int32 *restrict line1){
    int32 distances[4];

    if (segments_intersect(line0, line1))
        return 0;

    distances[0] = distance_point_segment(line0[0], line0[1], line1);
    distances[1] = distance_point_segment(line0[2], line0[3], line1);
    distances[2] = distance_point_segment(line1[0], line1[1], line0);
    distances[3] = distance_point_segment(line1[2], line1[3], line0);
    return minimum(distances);
}

int32
minimum(int32 const distances[4]) {
    int32 min = distances[0];
    for (int32 i = 1; i < 4; i += 1) {
        if (distances[i] < min)
            min = distances[i];
    }
    return min;
}

bool
segments_intersect(int32 *restrict line0, int32 *restrict line1) {
    int32 x0, y0, x1, y1;
    int32 xx0, yy0, xx1, yy1;
    int32 dx, dy, dxx, dyy;
    int32 delta;
    double s, t;
    bool ss, tt;

    x0 = line0[0]; y0 = line0[1];
    x1 = line0[2]; y1 = line0[3];
    xx0 = line1[0]; yy0 = line1[1];
    xx1 = line1[2]; yy1 = line1[3];
    
    dx = x1 - x0;
    dy = y1 - y0;
    dxx = xx1 - xx0;
    dyy = yy1 - yy0;

    delta = dxx*dy - dyy*dx;
    if (delta == 0)
        return false;

    s = (double) (dx*yy0 - y0 + dy*x0 - xx0) / (double) delta;
    t = (double) (dxx*y0 - yy0 + dyy*xx0 - x0) / (double) (-delta);

    ss = (0 <= s) && (s <= 1);
    tt = (0 <= t) && (t <= 1);
    return ss && tt;
}

int32
distance_point_segment(int32 const px, int32 const py, int32 *restrict line) {
    int32 x0, y0, x1, y1;
    int32 dx, dy;
    double distance;

    x0 = line[0]; y0 = line[1];
    x1 = line[2]; y1 = line[3];
    dx = x1 - x0;
    dy = y1 - y0;

    if ((dx == dy) && (dy == 0)) {
        dx = px - x0;
        dy = py - y0;
    } else {
        double t;
        t = (double) ((px - x0)*dx + (py - y0)*dy) / (double) (dx*dx + dy*dy);

        if (t < 0) {
            dx = px - x0;
            dy = py - y0;
        } else if (t > 1) {
            dx = px - x1;
            dy = py - y1;
        } else {
            double near_x = (double) x0 + t*dx;
            double near_y = (double) y0 + t*dy;
            dx = px - (int32) (round(near_x));
            dy = py - (int32) (round(near_y));
        }
    }
    distance = round(sqrt(dx*dx + dy*dy));
    return (int32) distance;
}

#ifndef TESTING_THIS_FILE
#define TESTING_THIS_FILE 1
#endif

#if TESTING_THIS_FILE

#define LINESIZE 4

#define Q(x) #x
#define QUOTE(x) Q(x)

#define PRINT_LINE(name) print_line(QUOTE(name), name)

static void
print_line(char *name, int *line) {
    printf("%s = [", name);
    for (int i = 0; i < (LINESIZE - 1); i += 1)
        printf("%d, ", line[i]);
    printf("%d]\n", line[LINESIZE - 1]);

    return;
}

int main(void) {
    struct timespec t0, t1;
    int ncalcs = 1000000;
    int distance = 0;
    int line0[LINESIZE];
    int line1[LINESIZE];

    for (int i = 0; i < LINESIZE; i += 1) {
        line0[i] = rand() % 512;
        line1[i] = rand() % 512;
    }

    PRINT_LINE(line0);
    PRINT_LINE(line1);

    clock_gettime(CLOCK_REALTIME, &t0);

    for (int i = 0; i < ncalcs; i += 1)
        distance += segments_distance(line0, line1);

    clock_gettime(CLOCK_REALTIME, &t1);

    printf("total distance: %d\n", distance);

    {
        long seconds = t1.tv_sec - t0.tv_sec;
        long nanos = t1.tv_nsec - t0.tv_nsec;
        double total_seconds = (double) seconds + (double) nanos/1.0e9;
        double per_calc = (total_seconds / ncalcs);
        printf("time elapsed for %d: %fs [%es / calculation]\n",
                ncalcs, total_seconds, per_calc);
    }
    exit(0);
}
#endif
