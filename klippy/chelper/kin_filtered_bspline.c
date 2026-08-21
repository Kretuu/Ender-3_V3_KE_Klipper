// Limited-preview filtered B-spline Klipper integration
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // floor, isfinite, sqrt
#include <stddef.h> // offsetof
#include <stdlib.h> // free, malloc
#include <string.h> // memcpy, memset
#include "compiler.h" // __visible
#include "filtered_bspline_internal.h"
#include "itersolve.h" // struct stepper_kinematics
#include "trapq.h" // struct move, struct trapq


/****************************************************************
 * Desired trajectory sampling and prepared batch storage
 ****************************************************************/

/** Evaluate the desired Cartesian TrapQ position at an absolute print time. */
static struct coord
get_desired_position(struct trapq *tq, double print_time)
{
    trapq_check_sentinels(tq);
    struct move *head = list_first_entry(&tq->moves, struct move, node);
    struct move *tail = list_last_entry(&tq->moves, struct move, node);
    struct move *move = list_next_entry(head, node);
    while (move != tail
           && print_time >= move->print_time + move->move_t)
        move = list_next_entry(move, node);
    if (move == tail)
        return tail->start_pos;
    if (print_time <= move->print_time)
        return move->start_pos;
    return move_get_coord(move, print_time - move->print_time);
}

static void
sample_desired_window(struct trapq *tq, double start_time,
                      double desired[FBF_AXIS_COUNT][FBF_PREVIEW_SAMPLES])
{
    // Preview times are monotonic.  Walk TrapQ once instead of restarting at
    // its head for every one of the 140 samples.
    trapq_check_sentinels(tq);
    struct move *head = list_first_entry(&tq->moves, struct move, node);
    struct move *tail = list_last_entry(&tq->moves, struct move, node);
    struct move *move = list_next_entry(head, node);
    int sample;
    for (sample = 0; sample < FBF_PREVIEW_SAMPLES; sample++) {
        double print_time = start_time + sample * FBF_SAMPLE_TIME;
        while (move != tail
               && print_time >= move->print_time + move->move_t)
            move = list_next_entry(move, node);
        struct coord position;
        if (move == tail)
            position = tail->start_pos;
        else if (print_time <= move->print_time)
            position = move->start_pos;
        else
            position = move_get_coord(move, print_time - move->print_time);
        desired[0][sample] = position.x;
        desired[1][sample] = position.y;
    }
}

/** Remove batches after command generation and hybrid training pass them. */
static void
discard_old_batches(struct fbf_controller *controller, double start_time)
{
    double discard_before = start_time - FBF_PREVIEW_TIME - FBF_BATCH_TIME;
    if (controller->mode == FBF_MODE_HYBRID) {
        int axis_index;
        for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++) {
            struct hybrid_snapshot *snapshot
                = &controller->axis[axis_index].applied_snapshot;
            if (snapshot->latest_sample_index < 0) {
                discard_before = controller->trajectory_start_time;
                break;
            }
            double measured_time = controller->trajectory_start_time
                + snapshot->latest_sample_index * FBF_SAMPLE_TIME;
            if (axis_index == 0 || measured_time < discard_before)
                discard_before = measured_time;
        }
    }
    while (!list_empty(&controller->batches)) {
        struct fbf_batch *batch = list_first_entry(
            &controller->batches, struct fbf_batch, node);
        if (batch->start_time + FBF_BATCH_TIME >= discard_before)
            break;
        int axis_index;
        for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++)
            if (controller->nominal_cache[axis_index] == batch)
                controller->nominal_cache[axis_index] = NULL;
        list_del(&batch->node);
        free(batch);
    }
}

/** Select coefficients, store their nominal response, and commit one batch. */
static int
prepare_batch(struct fbf_controller *controller, struct trapq *tq)
{
    double desired[FBF_AXIS_COUNT][FBF_PREVIEW_SAMPLES];
    double gamma[FBF_AXIS_COUNT][FBF_CURRENT_COEFFICIENTS];
    struct fbf_batch *batch = malloc(sizeof(*batch));
    if (!batch)
        return -1;

    sample_desired_window(tq, controller->next_batch_time, desired);
    int axis_index;
    for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++) {
        struct fbf_axis *axis = &controller->axis[axis_index];
        int hybrid_result = -1;
        if (controller->mode == FBF_MODE_HYBRID)
            hybrid_result = fbf_hybrid_solve_coefficients(
                controller, axis_index, controller->next_batch_time,
                desired[axis_index], gamma[axis_index]);
        if (hybrid_result) {
            if (controller->mode == FBF_MODE_HYBRID
                && axis->applied_snapshot.history_samples
                   >= controller->hybrid_warmup_samples)
                axis->solve_fallbacks++;
            fbf_solve_standard_coefficients(
                axis, desired[axis_index], gamma[axis_index]);
        }

        double nominal[FBF_PREVIEW_SAMPLES];
        fbf_calculate_nominal_prediction(
            axis, gamma[axis_index], nominal);
        memcpy(batch->nominal[axis_index], nominal,
               FBF_BATCH_SAMPLES * sizeof(double));
        fbf_reconstruct_command(
            axis, gamma[axis_index], batch->position[axis_index]);
    }

    int sample;
    for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++)
        for (sample = 0; sample <= FBF_BATCH_SAMPLES; sample++) {
            double position = batch->position[axis_index][sample];
            if (position < controller->position_min[axis_index]
                || position > controller->position_max[axis_index]) {
                free(batch);
                return -2;
            }
        }
    for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++)
        fbf_commit_coefficients(
            &controller->axis[axis_index], gamma[axis_index]);

    batch->start_time = controller->next_batch_time;
    list_add_tail(&batch->node, &controller->batches);
    controller->next_batch_time += FBF_BATCH_TIME;
    return 0;
}

static void
initialize_trajectory(struct fbf_controller *controller, struct trapq *tq,
                      double start_time)
{
    struct coord initial_position = get_desired_position(tq, start_time);
    int axis_index, history_index;
    for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++) {
        double value = initial_position.axis[axis_index];
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
            controller->axis[axis_index].history[history_index] = value;
        fbf_hybrid_reset_prediction(&controller->axis[axis_index]);
    }
    // Define t(k)=start_time+k*T_s for this continuous controller trajectory.
    controller->trajectory_start_time = start_time;
    controller->next_batch_time = start_time;
    controller->initialized = 1;
}


/****************************************************************
 * Public controller API
 ****************************************************************/

struct fbf_controller * __visible
filtered_bspline_controller_alloc(void)
{
    struct fbf_controller *controller = malloc(sizeof(*controller));
    if (!controller)
        return NULL;
    memset(controller, 0, sizeof(*controller));
    controller->generation = 1;
    controller->hybrid_warmup_samples = 5500;
    list_init(&controller->batches);
    return controller;
}

void __visible
filtered_bspline_controller_reset(struct fbf_controller *controller)
{
    memset(controller->nominal_cache, 0, sizeof(controller->nominal_cache));
    while (!list_empty(&controller->batches)) {
        struct fbf_batch *batch = list_first_entry(
            &controller->batches, struct fbf_batch, node);
        list_del(&batch->node);
        free(batch);
    }
    controller->initialized = 0;
    controller->generation++;
    // A new trajectory resets the numerical learner state.  Keep the previous
    // run's counters visible between ENABLE=0 and the next trajectory.
}

void __visible
filtered_bspline_controller_free(struct fbf_controller *controller)
{
    filtered_bspline_controller_reset(controller);
    free(controller);
}

int __visible
filtered_bspline_configure_axis(struct fbf_controller *controller, char axis,
                                int numerator_count, double numerator[],
                                int denominator_count, double denominator[])
{
    if (axis != 'x' && axis != 'y')
        return -1;
    return fbf_axis_configure_model(
        &controller->axis[axis - 'x'], numerator_count, numerator,
        denominator_count, denominator);
}

int __visible
filtered_bspline_set_axis_limits(struct fbf_controller *controller, char axis,
                                 double position_min, double position_max)
{
    if ((axis != 'x' && axis != 'y') || position_min >= position_max)
        return -1;
    int axis_index = axis - 'x';
    controller->position_min[axis_index] = position_min;
    controller->position_max[axis_index] = position_max;
    return 0;
}

int __visible
filtered_bspline_configure_hybrid(struct fbf_controller *controller,
                                  double regularization, int warmup_samples)
{
    if (!isfinite(regularization) || regularization <= 0.
        || warmup_samples < HYBRID_RESIDUAL_TERMS)
        return -1;
    controller->hybrid_warmup_samples = warmup_samples;
    return 0;
}

int __visible
filtered_bspline_set_mode(struct fbf_controller *controller, int mode)
{
    if (mode < FBF_MODE_DISABLED || mode > FBF_MODE_HYBRID)
        return -1;
    if (controller->mode == mode)
        return 0;
    filtered_bspline_controller_reset(controller);
    controller->mode = mode;
    return 0;
}

void __visible
filtered_bspline_set_enabled(struct fbf_controller *controller, int enabled)
{
    filtered_bspline_set_mode(
        controller, enabled ? FBF_MODE_STANDARD : FBF_MODE_DISABLED);
}

/** Publish one immutable learner result into reactor-owned prediction state. */
int __visible
filtered_bspline_apply_hybrid_snapshot(
    struct fbf_controller *controller, char axis, long long latest_sample_index,
    int history_samples, int training_samples, int measurement_errors,
    double weights[], double nominal_history[], double residual_history[])
{
    if ((axis != 'x' && axis != 'y') || latest_sample_index < 0
        || history_samples < 0 || training_samples < 0
        || measurement_errors < 0)
        return -1;
    struct fbf_axis *axis_data = &controller->axis[axis - 'x'];
    struct hybrid_snapshot *snapshot = &axis_data->applied_snapshot;
    if (latest_sample_index < snapshot->latest_sample_index)
        return 0;
    snapshot->latest_sample_index = latest_sample_index;
    snapshot->history_samples = history_samples;
    snapshot->training_samples = training_samples;
    snapshot->measurement_errors = measurement_errors;
    memcpy(snapshot->weights, weights, sizeof(snapshot->weights));
    memcpy(snapshot->nominal_history, nominal_history,
           sizeof(snapshot->nominal_history));
    memcpy(snapshot->residual_history, residual_history,
           sizeof(snapshot->residual_history));
    fbf_hybrid_invalidate_snapshot_caches(axis_data);
    return 0;
}

void __visible
filtered_bspline_clear_hybrid_snapshot(struct fbf_controller *controller,
                                       char axis)
{
    if (axis != 'x' && axis != 'y')
        return;
    struct fbf_axis *axis_data = &controller->axis[axis - 'x'];
    struct hybrid_snapshot *snapshot = &axis_data->applied_snapshot;
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->latest_sample_index = -1;
    fbf_hybrid_invalidate_snapshot_caches(axis_data);
}

int __visible
filtered_bspline_get_trajectory(struct fbf_controller *controller,
                                int *initialized,
                                double *trajectory_start_time)
{
    *initialized = controller->initialized;
    *trajectory_start_time = controller->trajectory_start_time;
    return 0;
}

int __visible
filtered_bspline_get_hybrid_status(
    struct fbf_controller *controller, char axis, int *history_samples,
    int *training_samples, int *active, int *measurement_errors,
    int *solve_fallbacks,
    int *hybrid_solves, int *prediction_gap_samples,
    int *maximum_prediction_gap_samples, int *history_replay_samples,
    int *maximum_history_replay_samples, double *weight_norm)
{
    if (axis != 'x' && axis != 'y')
        return -1;
    struct fbf_axis *axis_data = &controller->axis[axis - 'x'];
    struct hybrid_snapshot *snapshot = &axis_data->applied_snapshot;
    *history_samples = snapshot->history_samples;
    *training_samples = snapshot->training_samples;
    *active = snapshot->history_samples >= controller->hybrid_warmup_samples;
    *measurement_errors = snapshot->measurement_errors;
    *solve_fallbacks = axis_data->solve_fallbacks;
    *hybrid_solves = axis_data->hybrid_solves;
    *prediction_gap_samples = axis_data->prediction_gap_samples;
    *maximum_prediction_gap_samples
        = axis_data->maximum_prediction_gap_samples;
    *history_replay_samples = axis_data->history_replay_samples;
    *maximum_history_replay_samples
        = axis_data->maximum_history_replay_samples;
    double squared_norm = 0.;
    int index;
    for (index = 0; index < HYBRID_FEATURES; index++)
        squared_norm += snapshot->weights[index] * snapshot->weights[index];
    *weight_norm = sqrt(squared_norm);
    return 0;
}

/** Prepare every command batch for which a complete preview is available. */
int __visible
filtered_bspline_prepare(struct fbf_controller *controller, struct trapq *tq,
                         double start_time, double end_time, int is_final)
{
    if (controller->mode == FBF_MODE_DISABLED)
        return 0;
    if (!controller->axis[0].configured || !controller->axis[1].configured)
        return -1;
    if (!controller->initialized) {
        if (end_time <= start_time + FBF_TIME_EPSILON)
            return 0;
        initialize_trajectory(controller, tq, start_time);
    }

    discard_old_batches(controller, start_time);
    double latest_batch_start = is_final
        ? end_time : end_time - FBF_PREVIEW_TIME;
    while (controller->next_batch_time
           < latest_batch_start - FBF_TIME_EPSILON) {
        int result = prepare_batch(controller, tq);
        if (result)
            return result;
    }
    return 0;
}

static int
get_prepared_position(struct fbf_controller *controller, int axis_index,
                      double print_time, double *position)
{
    struct fbf_batch *batch;
    list_for_each_entry(batch, &controller->batches, node) {
        double relative_time = print_time - batch->start_time;
        if (relative_time < -FBF_TIME_EPSILON
            || relative_time > FBF_BATCH_TIME + FBF_TIME_EPSILON)
            continue;
        double sample_position = relative_time / FBF_SAMPLE_TIME;
        if (sample_position <= 0.) {
            *position = batch->position[axis_index][0];
            return 0;
        }
        if (sample_position >= FBF_BATCH_SAMPLES) {
            *position = batch->position[axis_index][FBF_BATCH_SAMPLES];
            return 0;
        }
        int sample = floor(sample_position);
        double fraction = sample_position - sample;
        double first = batch->position[axis_index][sample];
        double second = batch->position[axis_index][sample + 1];
        *position = first + fraction * (second - first);
        return 0;
    }
    return -1;
}

static int
get_cached_position(double start_time, double samples[], double print_time,
                    double *position)
{
    double relative_time = print_time - start_time;
    if (relative_time < -FBF_TIME_EPSILON
        || relative_time > FBF_BATCH_TIME + FBF_TIME_EPSILON)
        return -1;
    double sample_position = relative_time / FBF_SAMPLE_TIME;
    if (sample_position <= 0.) {
        *position = samples[0];
        return 0;
    }
    if (sample_position >= FBF_BATCH_SAMPLES) {
        *position = samples[FBF_BATCH_SAMPLES];
        return 0;
    }
    int sample = floor(sample_position);
    double fraction = sample_position - sample;
    *position = samples[sample]
        + fraction * (samples[sample + 1] - samples[sample]);
    return 0;
}

int __visible
filtered_bspline_get_position(struct fbf_controller *controller, char axis,
                              double print_time, double *position)
{
    if (axis != 'x' && axis != 'y')
        return -1;
    return get_prepared_position(
        controller, axis - 'x', print_time, position);
}


/****************************************************************
 * Stepper-kinematics wrapper
 ****************************************************************/

#define FBF_DUMMY_TIME 500.0

struct filtered_bspline_stepper {
    struct stepper_kinematics sk;
    struct stepper_kinematics *orig_sk;
    struct fbf_controller *controller;
    struct move dummy_move;
    unsigned int cached_generation;
    int cache_valid;
    double cached_start_time;
    double cached_position[FBF_AXIS_COUNT][FBF_BATCH_SAMPLES + 1];
};

static int
cache_prepared_batch(struct filtered_bspline_stepper *wrapper,
                     double print_time)
{
    struct fbf_controller *controller = wrapper->controller;
    struct fbf_batch *batch;
    list_for_each_entry(batch, &controller->batches, node) {
        double relative_time = print_time - batch->start_time;
        if (relative_time < -FBF_TIME_EPSILON
            || relative_time > FBF_BATCH_TIME + FBF_TIME_EPSILON)
            continue;
        wrapper->cached_start_time = batch->start_time;
        memcpy(wrapper->cached_position, batch->position,
               sizeof(wrapper->cached_position));
        wrapper->cached_generation = controller->generation;
        wrapper->cache_valid = 1;
        return 0;
    }
    wrapper->cache_valid = 0;
    return -1;
}

static double
filtered_bspline_calc_position(struct stepper_kinematics *sk,
                               struct move *move, double move_time)
{
    struct filtered_bspline_stepper *wrapper = container_of(
        sk, struct filtered_bspline_stepper, sk);
    struct fbf_controller *controller = wrapper->controller;
    if (controller->mode == FBF_MODE_DISABLED)
        return wrapper->orig_sk->calc_position_cb(
            wrapper->orig_sk, move, move_time);

    struct coord position = move_get_coord(move, move_time);
    double print_time = move->print_time + move_time;
    if (!wrapper->cache_valid
        || wrapper->cached_generation != controller->generation
        || print_time < wrapper->cached_start_time - FBF_TIME_EPSILON
        || print_time > wrapper->cached_start_time + FBF_BATCH_TIME
                        + FBF_TIME_EPSILON)
        cache_prepared_batch(wrapper, print_time);
    int found = 0;
    if (wrapper->cache_valid) {
        if (sk->active_flags & AF_X)
            found |= !get_cached_position(
                wrapper->cached_start_time, wrapper->cached_position[0],
                print_time, &position.x);
        if (sk->active_flags & AF_Y)
            found |= !get_cached_position(
                wrapper->cached_start_time, wrapper->cached_position[1],
                print_time, &position.y);
    }
    if (!found)
        return wrapper->orig_sk->calc_position_cb(
            wrapper->orig_sk, move, move_time);

    wrapper->dummy_move.start_pos = position;
    return wrapper->orig_sk->calc_position_cb(
        wrapper->orig_sk, &wrapper->dummy_move, FBF_DUMMY_TIME);
}

struct stepper_kinematics * __visible
filtered_bspline_stepper_alloc(struct fbf_controller *controller)
{
    struct filtered_bspline_stepper *wrapper = malloc(sizeof(*wrapper));
    if (!wrapper)
        return NULL;
    memset(wrapper, 0, sizeof(*wrapper));
    wrapper->controller = controller;
    wrapper->dummy_move.move_t = 2. * FBF_DUMMY_TIME;
    wrapper->sk.calc_position_cb = filtered_bspline_calc_position;
    return &wrapper->sk;
}

int __visible
filtered_bspline_stepper_set_sk(struct stepper_kinematics *sk,
                                struct stepper_kinematics *orig_sk)
{
    if (!(orig_sk->active_flags & (AF_X | AF_Y)))
        return -1;
    struct filtered_bspline_stepper *wrapper = container_of(
        sk, struct filtered_bspline_stepper, sk);
    wrapper->orig_sk = orig_sk;
    wrapper->sk.active_flags = orig_sk->active_flags;
    return 0;
}

void __visible
filtered_bspline_stepper_set_generation_window(
    struct stepper_kinematics *sk, double window)
{
    sk->gen_steps_pre_active = window;
    sk->gen_steps_post_active = window;
}
