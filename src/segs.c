#include <stdint.h>
#include <stdbool.h>
#include <math.h>

int32_t min(int32_t [4]);
int32_t point_seg_dist(int32_t px, int32_t py, int32_t *line);
bool segs_intersect(int32_t *line0, int32_t *line1);

int32_t segs(int32_t * restrict line0, int32_t * restrict line1){
    if (segs_intersect(line0, line1))
        return 0;
    int32_t distances[4];
    distances[0] = point_seg_dist(line0[0], line0[1], line1);
    distances[1] = point_seg_dist(line0[2], line0[3], line1);
    distances[2] = point_seg_dist(line1[0], line1[1], line0);
    distances[3] = point_seg_dist(line1[2], line1[3], line0);
    return min(distances);
}

bool segs_intersect(int32_t *line0, int32_t *line1) {
    double x0, y0, x1, y1;
    double xx0, yy0, xx1, yy1;
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

    double s = (dx0*(yy0 - y0) + dy0*(x0 - xx0)) / delta;
    double t = (dx1*(y0 - yy0) + dy1*(xx0 - x0)) / (-delta);

    bool ss = (0 <= s) && (s <= 1);
    bool tt = (0 <= t) && (t <= 1);
    return ss && tt;
}

int32_t point_seg_dist(int32_t px, int32_t py, int32_t *line) {
    double x0, y0, x1, y1;
    x0 = line[0];
    y0 = line[1];
    x1 = line[2];
    y1 = line[3];
    int dx = x1 - x0;
    int dy = y1 - y0;
    if ((dx == dy) && (dy == 0)) {
        dx = px - x0;
        dy = py - y0;
        goto calc_dist;
    }
    double t = ((px - x0)*dx + (py - y0)*dy) / (dx*dx + dy*dy);

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

int32_t min(int32_t distances[4]) {
    int i = 0;
    double m = distances[i];
    while (++i < 4) {
        if (distances[i] < m)
            m = distances[i];
    }
    return m;
}
