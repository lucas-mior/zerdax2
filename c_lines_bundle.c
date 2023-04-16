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

static const int32 min_angle = 15 * 100;
static const size_t line_size = LINE_FIELDS*sizeof(int32);
static int32 min_dist;

typedef struct Group {
    int32 lines[MAX_LINES_IN_GROUP][LINE_FIELDS];
    int32 angles[MAX_LINES_IN_GROUP];
    int32 len;
    struct Group *next;
} Group;

static Group *first = NULL;
static Group *last = NULL;

void *ealloc(void *old, size_t size) {
    void *p;
    if ((p = realloc(old, size)) == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes.\n", size);
        exit(1);
    }
    return p;
}

void *ecalloc(size_t nmemb, size_t size) {
    void *p;
    if ((p = calloc(nmemb, size)) == NULL) {
        fprintf(stderr, "Failed to allocate %zu members of %zu bytes each.\n", 
                        nmemb, size);
        exit(1);
    }
    return p;
}

static int32 compare(const void *a, const void *b) {
    int32 *c = (int32 *) a;
    int32 *d = (int32 *) b;
    return *c - *d;
}

static double median(int32 *array, int32 len) {
    qsort(array, len, sizeof(int32), compare);
    if ((len % 2) == 0) {
        return (double) (array[(int)((len/2) - 1)] + array[(int) (len/2)]) / (double) 2.0;
    } else {
        return (double) array[(int)(len/2)];
    }
}

static void append(Group *group, int32 line[LINE_FIELDS]) {
    int32 j = group->len;
    if (j >= MAX_LINES_IN_GROUP)
        return;

    group->angles[j] = line[I_ANGLE];
    memcpy(group->lines[j], line, line_size);
    group->len += 1;
    return;
}

static void groups_append(int32 line[LINE_FIELDS]) {
    Group *group = last;
    group->next = ealloc(NULL, sizeof(Group));
    group = group->next;
    group->next = NULL;
    group->len = 0;
    append(group, line);
    last = group;
    return;
}

static bool check_line_diff(int32 line1[LINE_FIELDS], Group *group) {
    int32 min_angle2 = min_angle + 2;
    while (group) {
        for (int i = 0; i < group->len; i += 1) {
            int32 *line0 = group->lines[i];
            int32 dtheta = abs(line1[I_ANGLE] - line0[I_ANGLE]);
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

int32 lines_bundle(int32 lines[][LINE_FIELDS], int32 bundled[][LINE_FIELDS], int32 n, int32 mdist) {
    min_dist = mdist;
    Group *group = ealloc(NULL, sizeof(Group));
    first = last = group;
    first->next = NULL;
    first->len = 0;
    append(group, lines[0]);

    for (int32 i = 1; i < n; i += 1) {
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
        int32 *best_line = group->lines[0];
        double min_diff = fabs(med - (double) best_line[I_ANGLE]);
        for (int i = 1; i < group->len; i += 1) {
            int32 *line = group->lines[i];
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
