#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "c_declarations.h"

#define MAX_LINES_IN_GROUP 8
#define LINE_FIELDS 6
#define I_ANGLE 5

static const int32 min_angle = 15 * 100;
static const size_t line_size = LINE_FIELDS*sizeof(int32);
static int32 min_distance;

typedef struct Group {
    int32 lines[MAX_LINES_IN_GROUP][LINE_FIELDS];
    int32 angles[MAX_LINES_IN_GROUP];
    int32 diff[MAX_LINES_IN_GROUP];
    int32 length;
    int32 unused;
    struct Group *next;
} Group;

static Group *first = NULL;
static Group *last = NULL;

int32 lines_bundle(int32 [][LINE_FIELDS],
                   int32 [][LINE_FIELDS],
                   int32, int32 );
static int32 compare(const void *, const void *);
static int32 median(int32 *, int32);
static void append(Group *, int32 [LINE_FIELDS]);
static void groups_append(int32 [LINE_FIELDS]);
static bool check_line_diff(int32 [LINE_FIELDS], Group *);

int32
lines_bundle(int32 lines[][LINE_FIELDS],
             int32 bundled[][LINE_FIELDS],
             int32 nlines, int32 min_distance0) {
    int m;
    Group *group = util_realloc(NULL, sizeof(*group));
    min_distance = min_distance0;
    first = last = group;
    first->next = NULL;
    first->length = 0;
    append(group, lines[0]);

    for (int32 i = 1; i < nlines; i += 1) {
        if (check_line_diff(lines[i], group))
            groups_append(lines[i]);
        group = first;
    }

    m = 0;
    memcpy(bundled[m], group->lines[0], line_size);
    group = group->next;
    m += 1;

    while (group) {
        int32 median_diff;
        int32 best_line[LINE_FIELDS] = {0};
        int32 number_bests = 0;
        int32 angle_median = median(group->angles, group->length);

        for (int i = 0; i < group->length; i += 1) {
            int32 *line = group->lines[i];
            group->diff[i] = abs(angle_median - line[I_ANGLE]);
        }
        median_diff = median(group->diff, group->length);

        for (int i = 0; i < group->length; i += 1) {
            int32 *line = group->lines[i];
            if (group->diff[i] <= median_diff) {
                for (int j = 0; j < LINE_FIELDS; j += 2) {
                    best_line[j] += line[j];
                    best_line[j+1] += line[j+1];
                }
                number_bests += 1;
            }
        }
        for (int j = 0; j < LINE_FIELDS; j += 1) {
            best_line[j] /= number_bests;
        }

        memcpy(bundled[m], best_line, line_size);
        {
            void *aux = group;
            group = group->next;
            free(aux);
        }
        m += 1;
    }
    return m;
}

int32
compare(const void *a, const void *b) {
    const int32 *c = a;
    const int32 *d = b;
    return *c - *d;
}

int32
median(int32 *array, int32 length) {
    qsort(array, (size_t) length, sizeof(*array), compare);
    if ((length % 2) == 0) {
        double result = round((array[(length/2) - 1] + array[length/2]) / 2.0);
        return (int32) result;
    } else {
        return array[length/2];
    }
}

void
append(Group *group, int32 line[LINE_FIELDS]) {
    int32 j = group->length;
    if (j >= MAX_LINES_IN_GROUP)
        return;

    group->angles[j] = line[I_ANGLE];
    memcpy(group->lines[j], line, line_size);
    group->length += 1;
    return;
}

void
groups_append(int32 line[LINE_FIELDS]) {
    Group *group = last;
    group->next = util_realloc(NULL, sizeof(*group));
    group = group->next;
    group->next = NULL;
    group->length = 0;
    append(group, line);
    last = group;
    return;
}

bool
check_line_diff(int32 line1[LINE_FIELDS], Group *group) {
    int32 min_angle2 = min_angle + 2*100;
    while (group) {
        for (int i = 0; i < group->length; i += 1) {
            int32 *line0 = group->lines[i];
            int32 dtheta = abs(line1[I_ANGLE] - line0[I_ANGLE]);
            if (dtheta < min_angle) {
                if (segments_distance(line0, line1) < min_distance) {
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
