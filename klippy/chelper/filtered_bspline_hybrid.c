// Online residual learning and delayed hybrid FBF prediction
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // fabs, isfinite, llround
#include <stdlib.h> // free, malloc
#include <string.h> // memcpy, memmove, memset
#include "compiler.h" // __visible
#include "filtered_bspline_internal.h"

/** Shift an oldest-to-newest scalar history and append the current sample. */
static void
shift_scalar_history(double history[], int count, double value)
{
    if (count > 1)
        memmove(history, history + 1, (count - 1) * sizeof(double));
    history[count - 1] = value;
}

/** Shift a history of coefficient vectors and append the current vector. */
static void
shift_coefficient_history(
    double history[][FBF_CURRENT_COEFFICIENTS], int count,
    const double value[FBF_CURRENT_COEFFICIENTS])
{
    if (count > 1)
        memmove(history, history + 1,
                (count - 1) * FBF_CURRENT_COEFFICIENTS * sizeof(double));
    memcpy(history[count - 1], value,
           FBF_CURRENT_COEFFICIENTS * sizeof(double));
}

/** Discard values derived from a superseded immutable learner snapshot. */
void
fbf_hybrid_invalidate_snapshot_caches(struct fbf_axis *axis)
{
    memset(&axis->prediction_cache, 0, sizeof(axis->prediction_cache));
    axis->prediction_cache.latest_sample_index = -1;
    axis->factorization_cache.valid = 0;
}

/** Remove the last complete learner snapshot from the motion controller. */
void
fbf_hybrid_reset_prediction(struct fbf_axis *axis)
{
    memset(&axis->applied_snapshot, 0, sizeof(axis->applied_snapshot));
    axis->applied_snapshot.latest_sample_index = -1;
    fbf_hybrid_invalidate_snapshot_caches(axis);
    axis->solve_fallbacks = axis->hybrid_solves = 0;
    axis->prediction_gap_samples = 0;
    axis->maximum_prediction_gap_samples = 0;
    axis->history_replay_samples = 0;
    axis->maximum_history_replay_samples = 0;
}


/****************************************************************
 * Worker-owned residual learner
 ****************************************************************/

/** Reset one worker-owned RLS state while preserving its plant model. */
static void
reset_training_axis(struct fbf_hybrid_learner *learner, int axis_index)
{
    struct hybrid_training_axis *axis = &learner->axis[axis_index];
    memset(axis->command_history, 0, sizeof(axis->command_history));
    memset(axis->covariance, 0, sizeof(axis->covariance));
    memset(&axis->snapshot, 0, sizeof(axis->snapshot));
    axis->snapshot.latest_sample_index = -1;
    axis->have_observation = 0;
    int row;
    for (row = 0; row < HYBRID_FEATURES; row++)
        axis->covariance[row][row] = 1. / learner->regularization;
}

/** Apply one recursive least-squares update without advancing 1 kHz state. */
static int
update_residual_weights(struct hybrid_training_axis *axis, double nominal,
                        double residual)
{
    struct hybrid_snapshot *snapshot = &axis->snapshot;
    // phi(k)=[1, yhat_pb(k-q+1...k), e_pb(k-p...k-1)]^T.
    double feature[HYBRID_FEATURES];
    feature[0] = 1.;
    int index;
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        feature[1 + index] = snapshot->nominal_history[index];
    feature[HYBRID_NOMINAL_TERMS] = nominal;
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        feature[1 + HYBRID_NOMINAL_TERMS + index]
            = snapshot->residual_history[index];

    // Reused RLS product: p_feature=P(k-1)*phi(k).
    double p_feature[HYBRID_FEATURES];
    int row, column;
    for (row = 0; row < HYBRID_FEATURES; row++) {
        double value = 0.;
        for (column = 0; column < HYBRID_FEATURES; column++)
            value += axis->covariance[row][column] * feature[column];
        p_feature[row] = value;
    }
    // RLS denominator: 1+phi(k)^T*P(k-1)*phi(k).
    double denominator = 1.;
    for (index = 0; index < HYBRID_FEATURES; index++)
        denominator += feature[index] * p_feature[index];
    if (!isfinite(denominator) || denominator <= 1.0e-12) {
        snapshot->measurement_errors++;
        return -1;
    }

    // Prediction error: e_pb(k)-w(k-1)^T*phi(k).
    double predicted = 0.;
    for (index = 0; index < HYBRID_FEATURES; index++)
        predicted += snapshot->weights[index] * feature[index];
    double prediction_error = residual - predicted;
    double inverse_denominator = 1. / denominator;
    double gain[HYBRID_FEATURES];
    // w(k)=w(k-1)+P*phi/(1+phi^T*P*phi)*prediction_error.
    for (index = 0; index < HYBRID_FEATURES; index++) {
        gain[index] = p_feature[index] * inverse_denominator;
        snapshot->weights[index]
            += gain[index] * prediction_error;
    }

    // P(k)=P(k-1)-(P*phi*phi^T*P)/(1+phi^T*P*phi).
    // P starts symmetric and this rank-one update preserves symmetry.  Update
    // one triangle and mirror it instead of calculating both halves.
    for (row = 0; row < HYBRID_FEATURES; row++) {
        for (column = row; column < HYBRID_FEATURES; column++) {
            double value = axis->covariance[row][column]
                - gain[row] * p_feature[column];
            axis->covariance[row][column] = value;
            axis->covariance[column][row] = value;
        }
    }

    snapshot->training_samples++;
    return 0;
}

/** Make one resampled value part of the past for the next 1 kHz sample. */
static int
advance_residual_history(struct hybrid_training_axis *axis, double nominal,
                         double residual, long long sample_index)
{
    struct hybrid_snapshot *snapshot = &axis->snapshot;
    if (sample_index != snapshot->latest_sample_index + 1) {
        snapshot->measurement_errors++;
        return -1;
    }
    shift_scalar_history(snapshot->nominal_history,
                         HYBRID_NOMINAL_TERMS - 1, nominal);
    shift_scalar_history(snapshot->residual_history,
                         HYBRID_RESIDUAL_TERMS, residual);
    snapshot->latest_sample_index = sample_index;
    snapshot->history_samples++;
    return 0;
}

struct fbf_hybrid_learner * __visible
filtered_bspline_learner_alloc(void)
{
    struct fbf_hybrid_learner *learner = malloc(sizeof(*learner));
    if (!learner)
        return NULL;
    memset(learner, 0, sizeof(*learner));
    learner->regularization = 0.01;
    reset_training_axis(learner, 0);
    reset_training_axis(learner, 1);
    return learner;
}

void __visible
filtered_bspline_learner_free(struct fbf_hybrid_learner *learner)
{
    free(learner);
}

int __visible
filtered_bspline_learner_configure_axis(
    struct fbf_hybrid_learner *learner, char axis, int numerator_count,
    double numerator[], int denominator_count, double denominator[])
{
    if (axis != 'x' && axis != 'y')
        return -1;
    struct hybrid_training_axis *axis_data = &learner->axis[axis - 'x'];
    if (fbf_build_impulse_response(
            axis_data->impulse, numerator_count, numerator,
            denominator_count, denominator))
        return -1;
    axis_data->configured = 1;
    return 0;
}

int __visible
filtered_bspline_learner_configure(struct fbf_hybrid_learner *learner,
                                   double regularization,
                                   int warmup_samples)
{
    if (!isfinite(regularization) || regularization <= 0.
        || warmup_samples < HYBRID_RESIDUAL_TERMS)
        return -1;
    learner->regularization = regularization;
    reset_training_axis(learner, 0);
    reset_training_axis(learner, 1);
    return 0;
}

static void
reset_learner_trajectory(struct fbf_hybrid_learner *learner,
                         unsigned int generation,
                         double trajectory_start_time)
{
    learner->generation = generation;
    learner->trajectory_start_time = trajectory_start_time;
    learner->initialized = 1;
    reset_training_axis(learner, 0);
    reset_training_axis(learner, 1);
}

/** Start a continuous observation segment without inventing missing data. */
static int
restart_observation_segment(struct fbf_hybrid_learner *learner,
                            int axis_index, double print_time,
                            double motor_position, double observed_position)
{
    struct hybrid_training_axis *axis = &learner->axis[axis_index];
    struct hybrid_snapshot *snapshot = &axis->snapshot;
    long long sample_index = floor(
        (print_time - learner->trajectory_start_time) / FBF_SAMPLE_TIME);
    if (sample_index < 0)
        return -1;

    int index;
    for (index = 0; index < FBF_IMPULSE_SAMPLES; index++)
        axis->command_history[index] = motor_position;
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        snapshot->nominal_history[index] = motor_position;
    memset(snapshot->residual_history, 0,
           sizeof(snapshot->residual_history));
    snapshot->residual_history[HYBRID_RESIDUAL_TERMS - 1]
        = observed_position - motor_position;
    snapshot->latest_sample_index = sample_index;
    axis->last_observation_time = print_time;
    axis->last_motor_position = motor_position;
    axis->last_observed_position = observed_position;
    axis->have_observation = 1;
    return 0;
}

/** Filter one committed command through the same finite plant model as FBF. */
static double
calculate_nominal_sample(struct hybrid_training_axis *axis,
                         double motor_position)
{
    shift_scalar_history(
        axis->command_history, FBF_IMPULSE_SAMPLES, motor_position);
    double nominal = 0.;
    int impulse_index;
    for (impulse_index = 0; impulse_index < FBF_IMPULSE_SAMPLES;
         impulse_index++)
        nominal += axis->impulse[impulse_index]
            * axis->command_history[FBF_IMPULSE_SAMPLES - 1 - impulse_index];
    return nominal;
}

/** Advance the model and residual learner on their shared 1 kHz grid. */
int __visible
filtered_bspline_learner_add_observation(
    struct fbf_hybrid_learner *learner, char axis, unsigned int generation,
    double trajectory_start_time, double print_time, double motor_position,
    double observed_position)
{
    if ((axis != 'x' && axis != 'y') || !generation
        || !isfinite(trajectory_start_time) || !isfinite(print_time)
        || !isfinite(motor_position) || !isfinite(observed_position))
        return -1;
    if (!learner->initialized || learner->generation != generation
        || fabs(learner->trajectory_start_time - trajectory_start_time)
           > FBF_TIME_EPSILON)
        reset_learner_trajectory(
            learner, generation, trajectory_start_time);
    if (print_time < trajectory_start_time - FBF_TIME_EPSILON)
        return 0;

    int axis_index = axis - 'x';
    struct hybrid_training_axis *axis_data = &learner->axis[axis_index];
    if (!axis_data->configured)
        return -1;
    if (!axis_data->have_observation)
        return restart_observation_segment(
            learner, axis_index, print_time, motor_position,
            observed_position);

    double interval = print_time - axis_data->last_observation_time;
    if (interval <= 0. || interval > HYBRID_MAX_OBSERVATION_GAP) {
        axis_data->snapshot.measurement_errors++;
        return restart_observation_segment(
            learner, axis_index, print_time, motor_position,
            observed_position);
    }

    long long last_index = floor(
        (print_time - trajectory_start_time) / FBF_SAMPLE_TIME
        + FBF_TIME_EPSILON);
    long long sample_index;
    for (sample_index = axis_data->snapshot.latest_sample_index + 1;
         sample_index <= last_index; sample_index++) {
        double sample_time = trajectory_start_time
            + sample_index * FBF_SAMPLE_TIME;
        if (sample_time <= axis_data->last_observation_time
            + FBF_TIME_EPSILON)
            continue;
        double fraction = (sample_time - axis_data->last_observation_time)
            / interval;
        if (fraction > 1. + FBF_TIME_EPSILON)
            break;
        double motor = axis_data->last_motor_position
            + fraction * (motor_position - axis_data->last_motor_position);
        double observed = axis_data->last_observed_position
            + fraction * (observed_position
                          - axis_data->last_observed_position);
        double nominal = calculate_nominal_sample(axis_data, motor);
        double residual = observed - nominal;
        if (update_residual_weights(axis_data, nominal, residual))
            return -2;
        if (advance_residual_history(
                axis_data, nominal, residual, sample_index))
            return -2;
    }
    axis_data->last_observation_time = print_time;
    axis_data->last_motor_position = motor_position;
    axis_data->last_observed_position = observed_position;
    return 0;
}

int __visible
filtered_bspline_learner_get_snapshot(
    struct fbf_hybrid_learner *learner, char axis,
    int *history_samples, int *training_samples, int *measurement_errors,
    long long *latest_sample_index, double weights[],
    double nominal_history[], double residual_history[])
{
    if (axis != 'x' && axis != 'y')
        return -1;
    struct hybrid_snapshot *snapshot
        = &learner->axis[axis - 'x'].snapshot;
    *history_samples = snapshot->history_samples;
    *training_samples = snapshot->training_samples;
    *measurement_errors = snapshot->measurement_errors;
    *latest_sample_index = snapshot->latest_sample_index;
    memcpy(weights, snapshot->weights, sizeof(snapshot->weights));
    memcpy(nominal_history, snapshot->nominal_history,
           sizeof(snapshot->nominal_history));
    memcpy(residual_history, snapshot->residual_history,
           sizeof(snapshot->residual_history));
    return 0;
}


/****************************************************************
 * Reactor-owned hybrid preview prediction
 ****************************************************************/

/** Retrieve the nominal prediction stored for one committed 1 kHz sample. */
int
fbf_hybrid_lookup_nominal_prediction(struct fbf_controller *controller,
                                     int axis_index, double print_time,
                                     double *nominal)
{
    struct list_node *node = controller->batches.root.next;
    struct fbf_batch *cached = controller->nominal_cache[axis_index];
    if (cached) {
        double cached_time = print_time - cached->start_time;
        if (cached_time >= -FBF_TIME_EPSILON
            && cached_time < FBF_BATCH_TIME - FBF_TIME_EPSILON) {
            long sample = llround(cached_time / FBF_SAMPLE_TIME);
            if (sample >= 0 && sample < FBF_BATCH_SAMPLES
                && fabs(cached_time - sample * FBF_SAMPLE_TIME) <= 1.0e-6) {
                *nominal = cached->nominal[axis_index][sample];
                return 0;
            }
        }
        if (print_time >= cached->start_time)
            node = cached->node.next;
    }
    while (node != &controller->batches.root) {
        struct fbf_batch *batch = container_of(
            node, struct fbf_batch, node);
        double relative_time = print_time - batch->start_time;
        long sample = llround(relative_time / FBF_SAMPLE_TIME);
        if (sample >= 0 && sample < FBF_BATCH_SAMPLES
            && fabs(relative_time - sample * FBF_SAMPLE_TIME) <= 1.0e-6) {
            controller->nominal_cache[axis_index] = batch;
            *nominal = batch->nominal[axis_index][sample];
            return 0;
        }
        if (relative_time < 0.)
            break;
        node = node->next;
    }
    return -1;
}

/** Evaluate ehat_pb(k)=w^T*phi(k) when every regressor value is numeric. */
static double
predict_scalar_residual(const double weights[HYBRID_FEATURES],
                        double nominal_history[], double nominal,
                        double residual_history[])
{
    // Bias weight.
    double value = weights[0];
    int index;
    // q-1 previous nominal predictions.
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        value += weights[1 + index] * nominal_history[index];
    // Current nominal prediction yhat_pb(k), completing the q terms.
    value += weights[HYBRID_NOMINAL_TERMS] * nominal;
    // p previous measured or recursively predicted residuals.
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        value += weights[1 + HYBRID_NOMINAL_TERMS + index]
            * residual_history[index];
    return value;
}

/** Predict the unmeasured residual gap up to the new preview window. */
static int
prepare_hybrid_history(struct fbf_controller *controller, int axis_index,
                       double start_time, double nominal_history[],
                       double residual_history[])
{
    struct fbf_axis *axis = &controller->axis[axis_index];
    struct hybrid_snapshot *snapshot = &axis->applied_snapshot;
    // k_0 is the first sample of the new FBF preview window.
    long long start_index = llround(
        (start_time - controller->trajectory_start_time) / FBF_SAMPLE_TIME);
    axis->history_replay_samples = 0;
    if (snapshot->latest_sample_index >= start_index)
        return -1;
    long long gap_samples = start_index - snapshot->latest_sample_index - 1;
    axis->prediction_gap_samples = gap_samples;
    if (gap_samples > axis->maximum_prediction_gap_samples)
        axis->maximum_prediction_gap_samples = gap_samples;

    struct hybrid_prediction_cache *cache = &axis->prediction_cache;
    if (!cache->valid || cache->latest_sample_index >= start_index) {
        // A newly adopted snapshot starts with measured residual history.
        memcpy(cache->nominal_history, snapshot->nominal_history,
               sizeof(cache->nominal_history));
        memcpy(cache->residual_history, snapshot->residual_history,
               sizeof(cache->residual_history));
        cache->latest_sample_index = snapshot->latest_sample_index;
        cache->valid = 1;
    }
    // Recursively predict ehat_pb for every unmeasured sample before k_0.
    long long sample_index;
    for (sample_index = cache->latest_sample_index + 1;
         sample_index < start_index; sample_index++) {
        double nominal;
        double print_time = controller->trajectory_start_time
            + sample_index * FBF_SAMPLE_TIME;
        // These commands were committed earlier, so yhat_pb(k) is numeric.
        if (fbf_hybrid_lookup_nominal_prediction(
                controller, axis_index, print_time, &nominal))
            return -1;
        // Calculate one ehat_pb(k), then make it a past value for k+1.
        double residual = predict_scalar_residual(
            snapshot->weights, cache->nominal_history, nominal,
            cache->residual_history);
        shift_scalar_history(cache->nominal_history,
                             HYBRID_NOMINAL_TERMS - 1, nominal);
        shift_scalar_history(cache->residual_history,
                             HYBRID_RESIDUAL_TERMS, residual);
        cache->latest_sample_index = sample_index;
        axis->history_replay_samples++;
        if (axis->history_replay_samples
            > axis->maximum_history_replay_samples)
            axis->maximum_history_replay_samples
                = axis->history_replay_samples;
    }
    memcpy(nominal_history, cache->nominal_history,
           sizeof(cache->nominal_history));
    memcpy(residual_history, cache->residual_history,
           sizeof(cache->residual_history));
    return 0;
}

/** Accumulate destination += scale * source for two coefficient vectors. */
static void
add_scaled_coefficients(double destination[FBF_CURRENT_COEFFICIENTS],
                        const double source[FBF_CURRENT_COEFFICIENTS],
                        double scale)
{
    int column;
    for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
        destination[column] += scale * source[column];
}

/**
 * Build A_h(w), the linear map from current coefficients to hybrid output.
 *
 * Past nominal and residual values have zero derivatives with respect to the
 * new gamma_C.  Future derivatives are propagated through the fixed learned
 * residual model.  Consequently A_h depends only on w and the fixed current
 * B-spline basis; numeric histories and start_index do not enter this method.
 */
static void
build_hybrid_preview_matrix(
    const double weights[HYBRID_FEATURES],
    const double current_basis
        [FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double matrix[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS])
{
    double nominal_history[HYBRID_NOMINAL_TERMS - 1]
        [FBF_CURRENT_COEFFICIENTS] = {{0.}};
    double residual_history[HYBRID_RESIDUAL_TERMS]
        [FBF_CURRENT_COEFFICIENTS] = {{0.}};

    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        const double *nominal = current_basis[row];
        double residual[FBF_CURRENT_COEFFICIENTS] = {0.};
        int index;
        for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
            add_scaled_coefficients(
                residual, nominal_history[index],
                weights[1 + index]);
        add_scaled_coefficients(
            residual, nominal, weights[HYBRID_NOMINAL_TERMS]);
        for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
            add_scaled_coefficients(
                residual, residual_history[index],
                weights[1 + HYBRID_NOMINAL_TERMS + index]);

        int column;
        for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
            matrix[row][column] = nominal[column] + residual[column];
        shift_coefficient_history(
            nominal_history, HYBRID_NOMINAL_TERMS - 1, nominal);
        shift_coefficient_history(
            residual_history, HYBRID_RESIDUAL_TERMS, residual);
    }
}

/** Factor A_h once for the currently adopted immutable weight snapshot. */
static int
ensure_hybrid_factorization(struct fbf_axis *axis)
{
    struct hybrid_preview_factorization *factorization
        = &axis->factorization_cache;
    if (factorization->valid)
        return 0;
    build_hybrid_preview_matrix(
        axis->applied_snapshot.weights, axis->filtered_current_basis,
        factorization->q);
    if (fbf_factorize_preview_matrix_qr(
            factorization->q, factorization->r))
        return -1;
    factorization->valid = 1;
    return 0;
}

/** Calculate b_pb(k), the nominal output fixed by committed coefficients. */
static double
calculate_nominal_offset(
    const double committed_coefficients[FBF_HISTORY_COEFFICIENTS],
    const double basis_row[FBF_HISTORY_COEFFICIENTS])
{
    double nominal = 0.;
    int history_index;
    for (history_index = 0;
         history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
        nominal += committed_coefficients[history_index]
            * basis_row[history_index];
    return nominal;
}

/**
 * Build rhs=r-b_h for one batch from its numeric past histories.
 *
 * Unlike A_h, b_h changes when start_index or committed coefficients change,
 * so this inexpensive scalar recursion is intentionally evaluated every batch.
 */
static void
build_hybrid_preview_rhs(
    const double weights[HYBRID_FEATURES],
    const double committed_coefficients[FBF_HISTORY_COEFFICIENTS],
    const double history_basis
        [FBF_PREVIEW_SAMPLES][FBF_HISTORY_COEFFICIENTS],
    const double desired[],
    double nominal_history[], double residual_history[], double rhs[])
{
    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        double nominal = calculate_nominal_offset(
            committed_coefficients, history_basis[row]);
        double residual = predict_scalar_residual(
            weights, nominal_history, nominal, residual_history);
        rhs[row] = desired[row] - nominal - residual;
        shift_scalar_history(
            nominal_history, HYBRID_NOMINAL_TERMS - 1, nominal);
        shift_scalar_history(
            residual_history, HYBRID_RESIDUAL_TERMS, residual);
    }
}

/** Prepare numeric history, build rhs, and solve with cached QR(A_h). */
int
fbf_hybrid_solve_coefficients(struct fbf_controller *controller,
                              int axis_index, double start_time,
                              double desired[], double gamma[])
{
    struct fbf_axis *axis = &controller->axis[axis_index];
    struct hybrid_snapshot *snapshot = &axis->applied_snapshot;
    // Signal the caller to use standard FBF until enough samples train w.
    if (snapshot->history_samples < controller->hybrid_warmup_samples)
        return -1;
    if (ensure_hybrid_factorization(axis))
        return -1;

    // Align numeric histories with the first sample of this preview window.
    double nominal_scalar[HYBRID_NOMINAL_TERMS - 1];
    double residual_scalar[HYBRID_RESIDUAL_TERMS];
    if (prepare_hybrid_history(controller, axis_index, start_time,
                               nominal_scalar, residual_scalar))
        return -1;

    double rhs[FBF_PREVIEW_SAMPLES];
    build_hybrid_preview_rhs(
        snapshot->weights, axis->history, axis->filtered_history_basis,
        desired, nominal_scalar, residual_scalar, rhs);

    // Solve gamma_C=argmin ||A_h*gamma_C-(y_d-b_h)||_2 by QR.
    struct hybrid_preview_factorization *factorization
        = &axis->factorization_cache;
    fbf_solve_qr(factorization->q, factorization->r, rhs, gamma);
    axis->hybrid_solves++;
    return 0;
}
