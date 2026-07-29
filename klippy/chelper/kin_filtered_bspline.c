// Limited-preview filtered B-spline feedforward kinematics
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // fabs, floor, sqrt
#include <stddef.h> // offsetof
#include <stdlib.h> // free, malloc
#include <string.h> // memset
#include "compiler.h" // __visible
#include "itersolve.h" // struct stepper_kinematics
#include "list.h" // list_head
#include "trapq.h" // struct move, struct trapq


/****************************************************************
 * Dissertation controller constants
 ****************************************************************/

// These values reproduce the nominal controller design used in the
// dissertation.  They are deliberately not configurable at runtime.
#define FBF_SAMPLE_TIME 0.001
#define FBF_DEGREE 5
#define FBF_KNOT_SPACING 10
#define FBF_BATCH_SAMPLES 70
#define FBF_PREVIEW_SAMPLES 140
// L_H = 20 satisfies L_C >= L_H + (n_up + degree)L for this design.
#define FBF_IMPULSE_SAMPLES 20

#define FBF_AXIS_COUNT 2
#define FBF_CURRENT_COEFFICIENTS \
    (FBF_PREVIEW_SAMPLES / FBF_KNOT_SPACING)
#define FBF_COMMITTED_COEFFICIENTS \
    (FBF_BATCH_SAMPLES / FBF_KNOT_SPACING)
#define FBF_HISTORY_COEFFICIENTS \
    (FBF_DEGREE + FBF_IMPULSE_SAMPLES / FBF_KNOT_SPACING)
#define FBF_TOTAL_COEFFICIENTS \
    (FBF_HISTORY_COEFFICIENTS + FBF_CURRENT_COEFFICIENTS)
#define FBF_BASIS_INPUT_SAMPLES \
    (FBF_PREVIEW_SAMPLES + FBF_IMPULSE_SAMPLES - 1)
#define FBF_BATCH_TIME (FBF_BATCH_SAMPLES * FBF_SAMPLE_TIME)
#define FBF_PREVIEW_TIME (FBF_PREVIEW_SAMPLES * FBF_SAMPLE_TIME)
#define FBF_TIME_EPSILON 1.0e-9

struct fbf_axis {
    int configured;
    double impulse[FBF_IMPULSE_SAMPLES];
    double filtered_history_basis
        [FBF_PREVIEW_SAMPLES][FBF_HISTORY_COEFFICIENTS];
    // Holds the current filtered basis during setup and Q afterwards.
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS];
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS];
    double history[FBF_HISTORY_COEFFICIENTS];
};

struct fbf_batch {
    double start_time;
    double position[FBF_AXIS_COUNT][FBF_BATCH_SAMPLES + 1];
    struct list_node node;
};

struct fbf_controller {
    int enabled, initialized;
    double next_batch_time;
    double position_min[FBF_AXIS_COUNT], position_max[FBF_AXIS_COUNT];
    struct fbf_axis axis[FBF_AXIS_COUNT];
    struct list_head batches;
};


/****************************************************************
 * B-spline and filtered-basis construction
 ****************************************************************/

/**
 * Evaluate one causal cardinal B-spline.
 *
 * The coefficient associated with sample zero has support from zero to
 * (degree + 1) knot intervals.  Extending the coefficient sequence into
 * negative time is equivalent to the constant pre-padding initialization
 * described by Duan et al. for the start of a trajectory.
 */
static double
cardinal_bspline(int degree, double sample)
{
    if (!degree)
        return sample >= 0. && sample < 1. ? 1. : 0.;
    if (sample <= 0. || sample >= degree + 1.)
        return 0.;
    return (sample / degree) * cardinal_bspline(degree - 1, sample)
        + ((degree + 1. - sample) / degree)
          * cardinal_bspline(degree - 1, sample - 1.);
}

/** Return an unfiltered basis value at an integer sample offset. */
static double
basis_value(int sample_offset)
{
    double knot_sample = sample_offset / (double)FBF_KNOT_SPACING;
    return cardinal_bspline(FBF_DEGREE, knot_sample);
}

/**
 * Construct the extended unfiltered basis matrix for one local preview.
 *
 * Columns -7,...,-1 represent retained history and columns 0,...,13
 * represent the coefficients optimized in the current window.  Its rows span
 * samples -19,...,139.  The -19,...,-1 convolution halo is not a past output
 * block: those are earlier unfiltered inputs whose plant responses extend
 * forward into current output samples 0,...,139.
 */
static void
build_extended_unfiltered_basis_matrix(
    double extended_basis
        [FBF_BASIS_INPUT_SAMPLES][FBF_TOTAL_COEFFICIENTS])
{
    int input_row, column;
    for (input_row = 0; input_row < FBF_BASIS_INPUT_SAMPLES; input_row++) {
        int sample = input_row - (FBF_IMPULSE_SAMPLES - 1);
        for (column = 0; column < FBF_TOTAL_COEFFICIENTS; column++) {
            int relative_coefficient = column - FBF_HISTORY_COEFFICIENTS;
            extended_basis[input_row][column] = basis_value(
                sample - relative_coefficient * FBF_KNOT_SPACING);
        }
    }
}

/**
 * Filter the extended basis and retain the current output block row.
 *
 * This explicitly implements filtered_local_basis = G_pb * extended_basis.
 * Each output entry is one sample of causal convolution.  Although the input
 * includes samples -19,...,-1, the retained output rows are only 0,...,139:
 *
 *     filtered_local_basis = [tilde_Phi_PC  tilde_Phi_C].
 */
static void
filter_local_basis_matrix(
    const struct fbf_axis *axis,
    const double extended_basis
        [FBF_BASIS_INPUT_SAMPLES][FBF_TOTAL_COEFFICIENTS],
    double filtered_local_basis
        [FBF_PREVIEW_SAMPLES][FBF_TOTAL_COEFFICIENTS])
{
    int row, column;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        for (column = 0; column < FBF_TOTAL_COEFFICIENTS; column++) {
            double value = 0.;
            int impulse_index;
            for (impulse_index = 0;
                 impulse_index < FBF_IMPULSE_SAMPLES; impulse_index++) {
                int input_row = row - impulse_index
                    + FBF_IMPULSE_SAMPLES - 1;
                value += axis->impulse[impulse_index]
                    * extended_basis[input_row][column];
            }
            filtered_local_basis[row][column] = value;
        }
    }
}

/**
 * Generate and DC-normalize the finite plant impulse response.
 *
 * The supplied coefficients use the usual z^-1 difference-equation order.
 * The truncated response is scaled to unity DC gain because a stationary
 * stepper position must map to the same stationary tool position.
 */
static int
build_impulse_response(struct fbf_axis *axis, int numerator_count
                       , double numerator[], int denominator_count
                       , double denominator[])
{
    if (numerator_count < 1 || denominator_count < 1
        || fabs(denominator[0]) < 1.0e-15)
        return -1;

    double inverse_a0 = 1. / denominator[0];
    int sample;
    for (sample = 0; sample < FBF_IMPULSE_SAMPLES; sample++) {
        double value = sample < numerator_count
            ? numerator[sample] * inverse_a0 : 0.;
        int order;
        for (order = 1; order < denominator_count && order <= sample;
             order++)
            value -= denominator[order] * inverse_a0
                * axis->impulse[sample - order];
        axis->impulse[sample] = value;
    }

    double truncated_dc_gain = 0.;
    for (sample = 0; sample < FBF_IMPULSE_SAMPLES; sample++)
        truncated_dc_gain += axis->impulse[sample];
    if (fabs(truncated_dc_gain) < 1.0e-12)
        return -1;
    for (sample = 0; sample < FBF_IMPULSE_SAMPLES; sample++)
        axis->impulse[sample] /= truncated_dc_gain;
    return 0;
}

/**
 * Split the current output block row into its past and current columns.
 *
 * The supplied matrix is [tilde_Phi_PC  tilde_Phi_C], not the complete
 * conceptual past/current/future matrix.  The first seven columns correspond
 * to coefficients retained from the preceding window.  The remaining fourteen
 * columns are the current unknowns and are copied to q before its in-place QR
 * factorization.
 */
static void
partition_filtered_basis(
    struct fbf_axis *axis,
    const double filtered_local_basis
        [FBF_PREVIEW_SAMPLES][FBF_TOTAL_COEFFICIENTS])
{
    int row, column;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        for (column = 0; column < FBF_HISTORY_COEFFICIENTS; column++)
            axis->filtered_history_basis[row][column]
                = filtered_local_basis[row][column];
        for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
            axis->q[row][column]
                = filtered_local_basis
                    [row][FBF_HISTORY_COEFFICIENTS + column];
    }
}

/**
 * Factorize the current filtered-basis matrix stored in q.
 *
 * Modified Gram-Schmidt overwrites the input matrix with the orthonormal Q
 * columns and stores the upper-triangular factor in r.  QR avoids explicitly
 * forming (Phi^T Phi)^-1 while solving the dissertation least-squares problem.
 */
static int
factorize_current_basis_qr(struct fbf_axis *axis)
{
    memset(axis->r, 0, sizeof(axis->r));

    int k, j;
    for (j = 0; j < FBF_CURRENT_COEFFICIENTS; j++) {
        // Until normalization, q[:, j] is the working vector w_j.
        int i;
        for (i = 0; i < j; i++) {
            double r_ij = 0.;
            for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
                r_ij += axis->q[k][i] * axis->q[k][j];
            axis->r[i][j] = r_ij;
            for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
                axis->q[k][j] -= r_ij * axis->q[k][i];
        }

        double r_jj = 0.;
        for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
            r_jj += axis->q[k][j] * axis->q[k][j];
        r_jj = sqrt(r_jj);
        if (r_jj < 1.0e-10)
            return -1;
        axis->r[j][j] = r_jj;
        for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
            axis->q[k][j] /= r_jj;
    }
    return 0;
}

/**
 * Build, filter, partition, and factorize the basis for one configured axis.
 *
 * These deliberately separate stages mirror the dissertation derivation.
 * They run once during axis configuration; only the resulting history basis,
 * Q, and R matrices are retained for preview-window calculations.
 */
static int
build_filtered_basis_qr(struct fbf_axis *axis)
{
    double extended_unfiltered_basis
        [FBF_BASIS_INPUT_SAMPLES][FBF_TOTAL_COEFFICIENTS];
    double filtered_local_basis
        [FBF_PREVIEW_SAMPLES][FBF_TOTAL_COEFFICIENTS];

    build_extended_unfiltered_basis_matrix(extended_unfiltered_basis);
    filter_local_basis_matrix(axis, extended_unfiltered_basis,
                              filtered_local_basis);
    partition_filtered_basis(axis, filtered_local_basis);
    return factorize_current_basis_qr(axis);
}


/****************************************************************
 * Window optimization
 ****************************************************************/

/** Solve R*gamma = Q^T*rhs for one axis and one preview window. */
static void
solve_coefficients(struct fbf_axis *axis, double desired[]
                   , double gamma[])
{
    double rhs[FBF_PREVIEW_SAMPLES];
    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        double past_output = 0.;
        int history_index;
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++) {
            past_output += axis->history[history_index]
                * axis->filtered_history_basis[row][history_index];
        }
        rhs[row] = desired[row] - past_output;
    }

    int column;
    for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++) {
        gamma[column] = 0.;
        for (row = 0; row < FBF_PREVIEW_SAMPLES; row++)
            gamma[column] += axis->q[row][column] * rhs[row];
    }
    for (column = FBF_CURRENT_COEFFICIENTS - 1; column >= 0; column--) {
        int next;
        for (next = column + 1; next < FBF_CURRENT_COEFFICIENTS; next++)
            gamma[column] -= axis->r[column][next] * gamma[next];
        gamma[column] /= axis->r[column][column];
    }
}

/** Reconstruct the committed command batch using the unfiltered basis. */
static void
reconstruct_command(struct fbf_axis *axis, double gamma[], double command[])
{
    int sample;
    for (sample = 0; sample <= FBF_BATCH_SAMPLES; sample++) {
        double value = 0.;
        int history_index;
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++) {
            int relative_coefficient = history_index
                - FBF_HISTORY_COEFFICIENTS;
            value += axis->history[history_index]
                * basis_value(
                    sample - relative_coefficient * FBF_KNOT_SPACING);
        }
        int column;
        for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
            value += gamma[column]
                * basis_value(sample - column * FBF_KNOT_SPACING);
        command[sample] = value;
    }
}

/** Retain only the coefficients that become past data in the next window. */
static void
commit_coefficients(struct fbf_axis *axis, double gamma[])
{
    int retained = FBF_HISTORY_COEFFICIENTS
        - FBF_COMMITTED_COEFFICIENTS;
    if (retained > 0)
        memmove(axis->history, axis->history + FBF_COMMITTED_COEFFICIENTS,
                retained * sizeof(axis->history[0]));
    int index;
    for (index = 0; index < FBF_COMMITTED_COEFFICIENTS; index++)
        axis->history[retained + index] = gamma[index];
}


/****************************************************************
 * Desired trajectory sampling and prepared batch storage
 ****************************************************************/

/** Evaluate the desired Cartesian trapq position at an absolute print time. */
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

/** Sample both desired axes over the current preview window. */
static void
sample_desired_window(struct trapq *tq, double start_time
                      , double desired[FBF_AXIS_COUNT][FBF_PREVIEW_SAMPLES])
{
    int sample;
    for (sample = 0; sample < FBF_PREVIEW_SAMPLES; sample++) {
        struct coord position = get_desired_position(
            tq, start_time + sample * FBF_SAMPLE_TIME);
        desired[0][sample] = position.x;
        desired[1][sample] = position.y;
    }
}

/** Remove command batches that are safely behind the step generator. */
static void
discard_old_batches(struct fbf_controller *controller, double start_time)
{
    double discard_before = start_time - FBF_PREVIEW_TIME - FBF_BATCH_TIME;
    while (!list_empty(&controller->batches)) {
        struct fbf_batch *batch = list_first_entry(
            &controller->batches, struct fbf_batch, node);
        if (batch->start_time + FBF_BATCH_TIME >= discard_before)
            break;
        list_del(&batch->node);
        free(batch);
    }
}

/** Optimize, reconstruct, and store one command batch. */
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
        solve_coefficients(axis, desired[axis_index], gamma[axis_index]);
        reconstruct_command(axis, gamma[axis_index],
                            batch->position[axis_index]);
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
        commit_coefficients(&controller->axis[axis_index], gamma[axis_index]);

    batch->start_time = controller->next_batch_time;
    list_add_tail(&batch->node, &controller->batches);
    controller->next_batch_time += FBF_BATCH_TIME;
    return 0;
}

/** Initialize the constant pre-history used at the trajectory boundary. */
static void
initialize_trajectory(struct fbf_controller *controller, struct trapq *tq
                      , double start_time)
{
    struct coord initial_position = get_desired_position(tq, start_time);
    int axis_index, history_index;
    for (axis_index = 0; axis_index < FBF_AXIS_COUNT; axis_index++) {
        double value = initial_position.axis[axis_index];
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
            controller->axis[axis_index].history[history_index] = value;
    }
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
    list_init(&controller->batches);
    return controller;
}

void __visible
filtered_bspline_controller_reset(struct fbf_controller *controller)
{
    while (!list_empty(&controller->batches)) {
        struct fbf_batch *batch = list_first_entry(
            &controller->batches, struct fbf_batch, node);
        list_del(&batch->node);
        free(batch);
    }
    controller->initialized = 0;
}

void __visible
filtered_bspline_controller_free(struct fbf_controller *controller)
{
    filtered_bspline_controller_reset(controller);
    free(controller);
}

int __visible
filtered_bspline_configure_axis(struct fbf_controller *controller, char axis
                                , int numerator_count, double numerator[]
                                , int denominator_count, double denominator[])
{
    if (axis != 'x' && axis != 'y')
        return -1;
    struct fbf_axis *axis_data = &controller->axis[axis - 'x'];
    memset(axis_data, 0, sizeof(*axis_data));
    if (build_impulse_response(axis_data, numerator_count, numerator,
                               denominator_count, denominator)
        || build_filtered_basis_qr(axis_data))
        return -1;
    axis_data->configured = 1;
    return 0;
}

int __visible
filtered_bspline_set_axis_limits(struct fbf_controller *controller, char axis
                                 , double position_min, double position_max)
{
    if ((axis != 'x' && axis != 'y') || position_min >= position_max)
        return -1;
    int axis_index = axis - 'x';
    controller->position_min[axis_index] = position_min;
    controller->position_max[axis_index] = position_max;
    return 0;
}

void __visible
filtered_bspline_set_enabled(struct fbf_controller *controller, int enabled)
{
    if (controller->enabled == !!enabled)
        return;
    filtered_bspline_controller_reset(controller);
    controller->enabled = !!enabled;
}

/**
 * Prepare every batch for which a complete preview is available.
 *
 * During normal queueing, the final 140 ms is retained as look-ahead.  During
 * a final flush, the trapq sentinel supplies a constant terminal position so
 * the remaining batches can be completed without inventing future motion.
 */
int __visible
filtered_bspline_prepare(struct fbf_controller *controller, struct trapq *tq
                         , double start_time, double end_time, int is_final)
{
    if (!controller->enabled)
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

/** Look up a linearly interpolated controller-grid position. */
static int
get_prepared_position(struct fbf_controller *controller, int axis_index
                      , double print_time, double *position)
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

int __visible
filtered_bspline_get_position(struct fbf_controller *controller, char axis
                              , double print_time, double *position)
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
};

/** Supply compensated Cartesian coordinates to the original kinematics. */
static double
filtered_bspline_calc_position(struct stepper_kinematics *sk, struct move *move
                               , double move_time)
{
    struct filtered_bspline_stepper *wrapper = container_of(
        sk, struct filtered_bspline_stepper, sk);
    struct fbf_controller *controller = wrapper->controller;
    if (!controller->enabled)
        return wrapper->orig_sk->calc_position_cb(
            wrapper->orig_sk, move, move_time);

    struct coord position = move_get_coord(move, move_time);
    double print_time = move->print_time + move_time;
    int found = 0;
    if (sk->active_flags & AF_X)
        found |= !get_prepared_position(
            controller, 0, print_time, &position.x);
    if (sk->active_flags & AF_Y)
        found |= !get_prepared_position(
            controller, 1, print_time, &position.y);
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
filtered_bspline_stepper_set_sk(struct stepper_kinematics *sk
                                , struct stepper_kinematics *orig_sk)
{
    if (!(orig_sk->active_flags & (AF_X | AF_Y)))
        return -1;
    struct filtered_bspline_stepper *wrapper = container_of(
        sk, struct filtered_bspline_stepper, sk);
    wrapper->orig_sk = orig_sk;
    wrapper->sk.active_flags = orig_sk->active_flags;
    return 0;
}
