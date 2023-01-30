#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

int32_t min(int32_t const distances[4]) {
    int32_t i = 0;
    int32_t m = distances[i];
    while (++i < 4) {
        if (distances[i] < m)
            m = distances[i];
    }
    return m;
}

bool segs_intersect(int32_t * restrict line0, int32_t * restrict line1) {
    int32_t x0, y0, x1, y1;
    int32_t xx0, yy0, xx1, yy1;
    x0 = line0[0];
    y0 = line0[1];
    x1 = line0[2];
    y1 = line0[3];
    xx0 = line1[0];
    yy0 = line1[1];
    xx1 = line1[2];
    yy1 = line1[3];
    
    int32_t dx0 = x1 - x0;
    int32_t dy0 = y1 - y0;
    int32_t dx1 = xx1 - xx0;
    int32_t dy1 = yy1 - yy0;

    int32_t delta = dx1*dy0 - dy1*dx0;
    if (delta == 0)
        return false;

    double s = ((double)dx0*((double)yy0 - (double)y0) + (double)dy0*((double)x0 - (double)xx0)) / (double)delta;
    double t = ((double)dx1*((double)y0 - (double)yy0) + (double)dy1*((double)xx0 - (double)x0)) / (double)(-delta);

    bool ss = (0 <= s) && (s <= 1);
    bool tt = (0 <= t) && (t <= 1);
    return ss && tt;
}

int32_t point_seg_dist(int32_t const px, int32_t const py, int32_t * restrict line) {
    int32_t x0, y0, x1, y1;
    x0 = line[0];
    y0 = line[1];
    x1 = line[2];
    y1 = line[3];
    int32_t dx = x1 - x0;
    int32_t dy = y1 - y0;
    if ((dx == dy) && (dy == 0)) {
        dx = px - x0;
        dy = py - y0;
        goto calc_dist;
    }
    double t = (((double)px - (double)x0)*(double)dx + ((double)py - (double)y0)*(double)dy) / ((double)dx*(double)dx + (double)dy*(double)dy);

    if (t < 0) {
        dx = px - x0;
        dy = py - y0;
    } else if (t > 1) {
        dx = px - x1;
        dy = py - y1;
    } else {
        double near_x = x0 + t*dx;
        double near_y = y0 + t*dy;
        dx = px - near_x;
        dy = py - near_y;
    }
    calc_dist:
    return sqrt(dx*dx + dy*dy);
}

int32_t segments_distance(int32_t * restrict line0, int32_t * restrict line1){
    if (segs_intersect(line0, line1))
        return 0;
    int32_t distances[4];
    distances[0] = point_seg_dist(line0[0], line0[1], line1);
    distances[1] = point_seg_dist(line0[2], line0[3], line1);
    distances[2] = point_seg_dist(line1[0], line1[1], line0);
    distances[3] = point_seg_dist(line1[2], line1[3], line0);
    return min(distances);
}
