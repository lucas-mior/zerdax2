#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "c_segments.h"

#define MAX_LINES_IN_GROUP 8
#define LINE_FIELDS 6
#define I_ANGLE 5
#define I_LENGTH 4

static const int32_t min_angle = 15 * 100;
static const size_t line_size = LINE_FIELDS*sizeof(int32_t);
static int32_t min_dist;

typedef struct Group {
    int32_t lines[MAX_LINES_IN_GROUP][LINE_FIELDS];
    int32_t angles[MAX_LINES_IN_GROUP];
    int32_t len;
    struct Group *next;
} Group;

static Group *first = NULL;
static Group *last = NULL;

static void *emalloc(size_t size) {
    void *p;
    if (!(p = malloc(size))) {
        fprintf(stderr, "Failed to allocate memory");
        exit(1);
    }
    return p;
}

static int32_t compare(const void *a, const void *b) {
    int32_t *c = (int32_t *) a;
    int32_t *d = (int32_t *) b;
    return *c - *d;
}

static double median(int32_t *array, int32_t len) {
    qsort(array, len, sizeof(int32_t), compare);
    if ((len % 2) == 0) {
        return (double) (array[(int)((len/2) - 1)] + array[(int) (len/2)]) / (double) 2.0;
    } else {
        return (double) array[(int)(len/2)];
    }
}

static void append(Group *group, int32_t line[LINE_FIELDS]) {
    int32_t j = group->len;
    if (j >= MAX_LINES_IN_GROUP)
        return;

    group->angles[j] = line[I_ANGLE];
    memcpy(group->lines[j], line, line_size);
    group->len += 1;
    return;
}

static void groups_append(int32_t line[LINE_FIELDS]) {
    Group *group = last;
    group->next = emalloc(sizeof(Group));
    group = group->next;
    group->next = NULL;
    group->len = 0;
    append(group, line);
    last = group;
    return;
}

static bool check_line_diff(int32_t line1[LINE_FIELDS], Group *group) {
    int32_t min_angle2 = min_angle + 2;
    while (group) {
        for (int i = 0; i < group->len; i += 1) {
            int32_t *line0 = group->lines[i];
            int32_t dtheta = abs(line1[I_ANGLE] - line0[I_ANGLE]);
            if (dtheta < min_angle) {
                if (segments_distance(line0, line1) < min_dist) {
                    append(group, line1);
                    return false;
                }
            } else if (dtheta <= min_angle2 || abs(dtheta - 180*100) < min_angle) {
                if (segments_distance(line0, line1) <= 1) {
                    append(group, line1);
                    return false;
                }
            }
        }
        group = group->next;
    }
    return true;
}

int32_t lines_bundle(int32_t lines[][LINE_FIELDS], int32_t bundled[][LINE_FIELDS], int32_t n, int32_t mdist) {
    min_dist = mdist;
    Group *group = emalloc(sizeof(Group));
    first = last = group;
    first->next = NULL;
    first->len = 0;
    append(group, lines[0]);

    for (int32_t i = 1; i < n; i += 1) {
        if (check_line_diff(lines[i], group))
            groups_append(lines[i]);
        group = first;
    }

    int m = 0;
    memcpy(bundled[m], group->lines[0], line_size);
    group = group->next;
    m += 1;
    while (group) {
        double med = median(group->angles, group->len);
        int32_t *best_line = group->lines[0];
        double min_diff = fabs(med - (double) best_line[I_ANGLE]);
        for (int i = 1; i < group->len; i += 1) {
            int32_t *line = group->lines[i];
            double diff = fabs(med - (double)line[I_ANGLE]);
            if (diff < min_diff) {
                best_line = line;
                min_diff = diff;
            } else if (diff == min_diff) {
                if (line[I_LENGTH] > best_line[I_LENGTH])
                    best_line = line;
            }
        }
        memcpy(bundled[m], best_line, line_size);
        void *aux = group;
        group = group->next;
        free(aux);
        m += 1;
    }
    return m;
}
