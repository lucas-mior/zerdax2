#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "c_segments.h"

static const int32_t min_dist = 15;
static const int32_t min_angle = 15;

typedef struct Group {
    int32_t line[6];
    struct Group *next;
} Group;

typedef struct Groups {
    struct Group *group;
    struct Groups *next;
} Groups;

static Groups *firsts = NULL;
static Groups *lasts = NULL;

static inline void copy_line(int32_t dest[6], int32_t src[6]) {
    for (int i = 0; i < 6; i += 1)
        dest[i] = src[i];
    return;
}

static Group *group_append(Group *group, int32_t line[6]) {
    group->next = malloc(sizeof(*group));
    if (!group->next) {
        fprintf(stderr, "Failed to allocate memory");
        exit(1);
    }
    group = group->next;
    copy_line(group->line, line);
    group->next = NULL;
    return group;
}

static void groups_append(int32_t line[6]) {
    Groups *groups = lasts;
    groups->next = malloc(sizeof(*groups));
    if (!groups->next) {
        fprintf(stderr, "Failed to allocate memory");
        exit(1);
    }
    groups = groups->next;

    groups->group = malloc(10*sizeof(Group));
    groups->group->next = NULL;

    copy_line(groups->group->line, line);
    groups->next = NULL;
    lasts = groups;
    return;
}

static bool check_line_diff(int32_t line1[6], Groups *groups) {
    int32_t min_angle2 = min_angle + 2;
    Group *group = groups->group;
    while (groups) {
        group = groups->group;
        while (group) {
            int32_t *line0 = group->line;
            int32_t dist;
            int32_t dtheta = abs(line1[5] - line0[5]);
            if (dtheta < min_angle) {
                dist = segments_distance(line0, line1);
                if (dist < min_dist) {
                    group_append(group, line1);
                    return false;
                }
            } else if (dtheta <= min_angle2 || abs(dtheta - 180) < min_angle) {
                dist = segments_distance(line0, line1);
                if (dist <= 1) {
                    group_append(group, line1);
                    return false;
                }
            }
            group = group->next;
        }
        groups = groups->next;
    }
    return true;
}

int32_t lines_bundle(int32_t lines[][6], int32_t bundled[][6], int32_t n) {
    Groups *groups = malloc(10*sizeof(Groups));
    firsts = lasts = groups;
    firsts->next = NULL;
    firsts->group = malloc(10*sizeof(Group));
    copy_line(firsts->group->line, lines[0]);

    for (int32_t i = 1; i < n; i += 1) {
        if (check_line_diff(lines[i], groups)) {
            groups_append(lines[i]);
        }
        groups = firsts;
    }
    groups = firsts;
    int m = 0;
    Group *group;
    while (groups) {
        group = groups->group;
        while (group) {
            if (group->line[4] > bundled[m][5])
                copy_line(bundled[m], group->line);
            group = group->next;
        }
        groups = groups->next;
        m += 1;
    }
    return m;
}
