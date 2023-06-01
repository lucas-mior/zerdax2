#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "c_declarations.h"

static int32 min(int32 const [4]);
static bool segments_intersect(int32 * restrict, int32 * restrict);
static int32 distance_point_segment(int32 const, int32 const, int32 * restrict);

int32 segments_distance(int32 * restrict line0, int32 * restrict line1){
    if (segments_intersect(line0, line1))
        return 0;
    int32 distances[4];
    distances[0] = distance_point_segment(line0[0], line0[1], line1);
    distances[1] = distance_point_segment(line0[2], line0[3], line1);
    distances[2] = distance_point_segment(line1[0], line1[1], line0);
    distances[3] = distance_point_segment(line1[2], line1[3], line0);
    return min(distances);
}

int32 min(int32 const distances[4]) {
    int32 i = 0;
    int32 m = distances[i];
    while (++i < 4) {
        if (distances[i] < m)
            m = distances[i];
    }
    return m;
}

bool segments_intersect(int32 * restrict line0, int32 * restrict line1) {
    int32 x0, y0, x1, y1;
    int32 xx0, yy0, xx1, yy1;
    x0 = line0[0]; y0 = line0[1];
    x1 = line0[2]; y1 = line0[3];
    xx0 = line1[0]; yy0 = line1[1];
    xx1 = line1[2]; yy1 = line1[3];
    
    int32 dx = x1 - x0;
    int32 dy = y1 - y0;
    int32 dxx = xx1 - xx0;
    int32 dyy = yy1 - yy0;

    int32 delta = dxx*dy - dyy*dx;
    if (delta == 0)
        return false;

    double s = (double) (dx*yy0 - y0 + dy*x0 - xx0) / (double)delta;
    double t = (double) (dxx*y0 - yy0 + dyy*xx0 - x0) / (double)(-delta);

    bool ss = (0 <= s) && (s <= 1);
    bool tt = (0 <= t) && (t <= 1);
    return ss && tt;
}

int32 distance_point_segment(int32 const px, int32 const py, int32 * restrict line) {
    int32 x0, y0, x1, y1;
    x0 = line[0]; y0 = line[1];
    x1 = line[2]; y1 = line[3];

    int32 dx = x1 - x0;
    int32 dy = y1 - y0;
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
            dx = px - near_x;
            dy = py - near_y;
        }
    }
    return sqrt(dx*dx + dy*dy);
}
