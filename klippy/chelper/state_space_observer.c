// Small discrete state-space observer used by the Trinkey acceleration stream
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // isfinite
#include <stdlib.h> // free, malloc
#include <string.h> // memcpy, memset
#include "compiler.h" // __visible
#include "state_space_observer.h"

#define OBSERVER_MAX_STATES 8

struct state_space_observer {
    int state_count, initialized;
    double fo[OBSERVER_MAX_STATES * OBSERVER_MAX_STATES];
    double gu[OBSERVER_MAX_STATES];
    double gv[OBSERVER_MAX_STATES];
    double ga[OBSERVER_MAX_STATES];
    double ho[OBSERVER_MAX_STATES];
    double x0_per_unit[OBSERVER_MAX_STATES];
    double state[OBSERVER_MAX_STATES];
    double previous_position;
    double previous_velocity;
    double previous_acceleration;
};

static int
all_finite(double values[], int count)
{
    int index;
    for (index = 0; index < count; index++)
        if (!isfinite(values[index]))
            return 0;
    return 1;
}

struct state_space_observer * __visible
state_space_observer_alloc(int state_count, double fo[], double gu[]
                           , double gv[], double ga[], double ho[]
                           , double x0_per_unit[])
{
    if (state_count < 1 || state_count > OBSERVER_MAX_STATES
        || !all_finite(fo, state_count * state_count)
        || !all_finite(gu, state_count)
        || !all_finite(gv, state_count)
        || !all_finite(ga, state_count)
        || !all_finite(ho, state_count)
        || !all_finite(x0_per_unit, state_count))
        return NULL;

    struct state_space_observer *observer = malloc(sizeof(*observer));
    if (!observer)
        return NULL;
    memset(observer, 0, sizeof(*observer));
    observer->state_count = state_count;
    memcpy(observer->fo, fo,
           state_count * state_count * sizeof(double));
    memcpy(observer->gu, gu, state_count * sizeof(double));
    memcpy(observer->gv, gv, state_count * sizeof(double));
    memcpy(observer->ga, ga, state_count * sizeof(double));
    memcpy(observer->ho, ho, state_count * sizeof(double));
    memcpy(observer->x0_per_unit, x0_per_unit,
           state_count * sizeof(double));
    return observer;
}

void __visible
state_space_observer_free(struct state_space_observer *observer)
{
    free(observer);
}

void __visible
state_space_observer_reset(struct state_space_observer *observer)
{
    observer->initialized = 0;
    memset(observer->state, 0, sizeof(observer->state));
}

/**
 * Return the position estimate at the current sample time and advance the
 * state using the inputs held since the preceding sample.
 */
double __visible
state_space_observer_sample(struct state_space_observer *observer
                            , double input_position, double input_velocity
                            , double measured_acceleration)
{
    int state_count = observer->state_count;
    if (!observer->initialized) {
        int index;
        for (index = 0; index < state_count; index++)
            observer->state[index]
                = observer->x0_per_unit[index] * input_position;
        observer->initialized = 1;
    } else {
        double next_state[OBSERVER_MAX_STATES];
        int row, column;
        for (row = 0; row < state_count; row++) {
            double value = 0.;
            for (column = 0; column < state_count; column++)
                value += observer->fo[row * state_count + column]
                       * observer->state[column];
            next_state[row] = value
                + observer->gu[row] * observer->previous_position
                + observer->gv[row] * observer->previous_velocity
                + observer->ga[row] * observer->previous_acceleration;
        }
        memcpy(observer->state, next_state,
               state_count * sizeof(double));
    }

    observer->previous_position = input_position;
    observer->previous_velocity = input_velocity;
    observer->previous_acceleration = measured_acceleration;

    double output = 0.;
    int index;
    for (index = 0; index < state_count; index++)
        output += observer->ho[index] * observer->state[index];
    return output;
}
