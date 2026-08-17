// Online residual learning and delayed hybrid FBF prediction
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // fabs, isfinite, llround
#include <string.h> // memcpy, memmove, memset
#include "filtered_bspline_internal.h"

/**
 * Constant and gamma_C-dependent parts of one future prediction:
 *
 *     yhat_pb(k) = offset + coefficient^T * gamma_C, or
 *     ehat_pb(k) = offset + coefficient^T * gamma_C.
 *
 * The offset stores the part already determined by past samples and committed
 * B-spline coefficients.  The coefficient vector stores how the prediction
 * depends on each current B-spline coefficient in the still-unknown gamma_C.
 *
 * Keeping future values in this form lets the residual predictor be evaluated
 * recursively without first knowing gamma_C.  Adding the nominal and residual
 * forms for every preview sample produces the corresponding row of
 * y_h = A_h * gamma_C + b_h used by the final least-squares problem.
 */
struct affine_value {
    double offset;
    double coefficient[FBF_CURRENT_COEFFICIENTS];
};

/** Shift an oldest-to-newest scalar history and append the current sample. */
static void
shift_scalar_history(double history[], int count, double value)
{
    if (count > 1)
        memmove(history, history + 1, (count - 1) * sizeof(double));
    history[count - 1] = value;
}

/** Shift an oldest-to-newest affine history and append one prediction. */
static void
shift_affine_history(struct affine_value history[], int count,
                     const struct affine_value *value)
{
    if (count > 1)
        memmove(history, history + 1,
                (count - 1) * sizeof(struct affine_value));
    history[count - 1] = *value;
}

/** Reset ridge RLS to w=0 and P=lambda^-1 I. */
void
fbf_hybrid_reset(struct fbf_axis *axis, double initial_position,
                 double regularization, int warmup_samples)
{
    struct hybrid_learner *learner = &axis->hybrid;
    memset(learner, 0, sizeof(*learner));
    learner->regularization = regularization;
    learner->warmup_samples = warmup_samples;
    learner->latest_sample_index = -1;
    // Ridge-RLS initial conditions: w(0)=0 and P(0)=lambda^-1*I.
    int row;
    for (row = 0; row < HYBRID_FEATURES; row++)
        learner->covariance[row][row] = 1. / regularization;
    for (row = 0; row < HYBRID_NOMINAL_TERMS - 1; row++)
        learner->nominal_history[row] = initial_position;
}

/** Apply one recursive least-squares update using a measured residual. */
int
fbf_hybrid_train_residual_sample(struct fbf_axis *axis, double nominal,
                                 double residual, long long sample_index)
{
    struct hybrid_learner *learner = &axis->hybrid;
    if (sample_index != learner->latest_sample_index + 1) {
        learner->measurement_errors++;
        return -1;
    }

    // phi(k)=[1, yhat_pb(k-q+1...k), e_pb(k-p...k-1)]^T.
    double feature[HYBRID_FEATURES];
    feature[0] = 1.;
    int index;
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        feature[1 + index] = learner->nominal_history[index];
    feature[HYBRID_NOMINAL_TERMS] = nominal;
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        feature[1 + HYBRID_NOMINAL_TERMS + index]
            = learner->residual_history[index];

    // Reused RLS product: p_feature=P(k-1)*phi(k).
    double p_feature[HYBRID_FEATURES];
    int row, column;
    for (row = 0; row < HYBRID_FEATURES; row++) {
        double value = 0.;
        for (column = 0; column < HYBRID_FEATURES; column++)
            value += learner->covariance[row][column] * feature[column];
        p_feature[row] = value;
    }
    // RLS denominator: 1+phi(k)^T*P(k-1)*phi(k).
    double denominator = 1.;
    for (index = 0; index < HYBRID_FEATURES; index++)
        denominator += feature[index] * p_feature[index];
    if (!isfinite(denominator) || denominator <= 1.0e-12) {
        learner->measurement_errors++;
        return -1;
    }

    // Prediction error: e_pb(k)-w(k-1)^T*phi(k).
    double predicted = 0.;
    for (index = 0; index < HYBRID_FEATURES; index++)
        predicted += learner->weights[index] * feature[index];
    double prediction_error = residual - predicted;
    // w(k)=w(k-1)+P*phi/(1+phi^T*P*phi)*prediction_error.
    for (index = 0; index < HYBRID_FEATURES; index++)
        learner->weights[index]
            += p_feature[index] / denominator * prediction_error;

    // P(k)=P(k-1)-(P*phi*phi^T*P)/(1+phi^T*P*phi).
    for (row = 0; row < HYBRID_FEATURES; row++)
        for (column = 0; column < HYBRID_FEATURES; column++)
            learner->covariance[row][column]
                -= p_feature[row] * p_feature[column] / denominator;

    // Advance phi(k)'s histories so this sample is "past" at k+1.
    shift_scalar_history(learner->nominal_history,
                         HYBRID_NOMINAL_TERMS - 1, nominal);
    shift_scalar_history(learner->residual_history,
                         HYBRID_RESIDUAL_TERMS, residual);
    learner->latest_sample_index = sample_index;
    learner->training_samples++;
    return 0;
}

/** Retrieve the nominal prediction stored for one committed 1 kHz sample. */
int
fbf_hybrid_lookup_nominal_prediction(struct fbf_controller *controller,
                                     int axis_index, double print_time,
                                     double *nominal)
{
    struct fbf_batch *batch;
    list_for_each_entry(batch, &controller->batches, node) {
        double relative_time = print_time - batch->start_time;
        long sample = llround(relative_time / FBF_SAMPLE_TIME);
        if (sample < 0 || sample >= FBF_BATCH_SAMPLES
            || fabs(relative_time - sample * FBF_SAMPLE_TIME) > 1.0e-6)
            continue;
        *nominal = batch->nominal[axis_index][sample];
        return 0;
    }
    return -1;
}

/** Evaluate ehat_pb(k)=w^T*phi(k) when every regressor value is numeric. */
static double
predict_scalar_residual(struct hybrid_learner *learner,
                        double nominal_history[], double nominal,
                        double residual_history[])
{
    // Bias weight.
    double value = learner->weights[0];
    int index;
    // q-1 previous nominal predictions.
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        value += learner->weights[1 + index] * nominal_history[index];
    // Current nominal prediction yhat_pb(k), completing the q terms.
    value += learner->weights[HYBRID_NOMINAL_TERMS] * nominal;
    // p previous measured or recursively predicted residuals.
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        value += learner->weights[1 + HYBRID_NOMINAL_TERMS + index]
            * residual_history[index];
    return value;
}

/** Predict the unmeasured residual gap up to the new preview window. */
static int
prepare_hybrid_history(struct fbf_controller *controller, int axis_index,
                       double start_time, double nominal_history[],
                       double residual_history[])
{
    struct hybrid_learner *learner = &controller->axis[axis_index].hybrid;
    // Start from the histories ending at the latest measured residual.
    memcpy(nominal_history, learner->nominal_history,
           sizeof(learner->nominal_history));
    memcpy(residual_history, learner->residual_history,
           sizeof(learner->residual_history));

    // k_0 is the first sample of the new FBF preview window.
    long long start_index = llround(
        (start_time - controller->trajectory_start_time) / FBF_SAMPLE_TIME);
    if (learner->latest_sample_index >= start_index)
        return -1;
    long long gap_samples = start_index - learner->latest_sample_index - 1;
    learner->prediction_gap_samples = gap_samples;
    if (gap_samples > learner->maximum_prediction_gap_samples)
        learner->maximum_prediction_gap_samples = gap_samples;
    // Recursively predict ehat_pb for every unmeasured sample before k_0.
    long long sample_index;
    for (sample_index = learner->latest_sample_index + 1;
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
            learner, nominal_history, nominal, residual_history);
        shift_scalar_history(nominal_history,
                             HYBRID_NOMINAL_TERMS - 1, nominal);
        shift_scalar_history(residual_history,
                             HYBRID_RESIDUAL_TERMS, residual);
    }
    return 0;
}

/**
 * Accumulate destination += scale * source for two affine values.
 *
 * A value represents either yhat_pb(k) or ehat_pb(k) as an offset plus a
 * coefficient vector multiplying gamma_C.  Scaling and adding it therefore
 * requires applying the same scale to both parts.  This is how one weighted
 * term of the learned regressor is added without yet knowing gamma_C.
 */
static void
add_scaled_affine(struct affine_value *destination,
                  const struct affine_value *source, double scale)
{
    destination->offset += scale * source->offset;
    int column;
    for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
        destination->coefficient[column]
            += scale * source->coefficient[column];
}

/** Convert the measured/predicted scalar past into affine constants. */
static void
initialize_affine_histories(
    double nominal_scalar[], double residual_scalar[],
    struct affine_value nominal_history[],
    struct affine_value residual_history[])
{
    // Past values are known: value=offset+0^T*gamma_C.
    memset(nominal_history, 0,
           (HYBRID_NOMINAL_TERMS - 1) * sizeof(struct affine_value));
    memset(residual_history, 0,
           HYBRID_RESIDUAL_TERMS * sizeof(struct affine_value));
    int index;
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        nominal_history[index].offset = nominal_scalar[index];
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        residual_history[index].offset = residual_scalar[index];
}

/** Build y_pb(k)=a_pb(k)'*gamma_C+b_pb(k) for one preview row. */
static void
build_nominal_affine(struct fbf_axis *axis, int row,
                     struct affine_value *nominal)
{
    memset(nominal, 0, sizeof(*nominal));
    // b_pb(k)=[tilde_Phi_PC](k,:)*gamma_P from committed coefficients.
    int history_index;
    for (history_index = 0;
         history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
        nominal->offset += axis->history[history_index]
            * axis->filtered_history_basis[row][history_index];
    // a_pb(k)^T=[tilde_Phi_C](k,:), multiplying the unknown gamma_C.
    int column;
    for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
        nominal->coefficient[column]
            = axis->filtered_current_basis[row][column];
}

/** Build e_hat(k)=a_e(k)'*gamma_C+b_e(k) from the learned regressor. */
static void
build_residual_affine(
    struct hybrid_learner *learner,
    struct affine_value nominal_history[],
    struct affine_value *nominal,
    struct affine_value residual_history[],
    struct affine_value *residual)
{
    memset(residual, 0, sizeof(*residual));
    // Regression bias w_0.
    residual->offset = learner->weights[0];
    int index;
    // Weighted yhat_pb(k-q+1...k-1) terms.
    for (index = 0; index < HYBRID_NOMINAL_TERMS - 1; index++)
        add_scaled_affine(residual, &nominal_history[index],
                          learner->weights[1 + index]);
    // Weighted current yhat_pb(k), completing the q nominal terms.
    add_scaled_affine(
        residual, nominal, learner->weights[HYBRID_NOMINAL_TERMS]);
    // Weighted e_pb(k-p...k-1), including earlier preview predictions.
    for (index = 0; index < HYBRID_RESIDUAL_TERMS; index++)
        add_scaled_affine(
            residual, &residual_history[index],
            learner->weights[1 + HYBRID_NOMINAL_TERMS + index]);
}

/** Assemble A_h and y_d-b_h by walking the recursive preview in time. */
static void
build_hybrid_preview_system(
    struct fbf_axis *axis, double desired[], double nominal_scalar[],
    double residual_scalar[],
    double matrix[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double rhs[FBF_PREVIEW_SAMPLES])
{
    struct hybrid_learner *learner = &axis->hybrid;
    struct affine_value nominal_history[HYBRID_NOMINAL_TERMS - 1];
    struct affine_value residual_history[HYBRID_RESIDUAL_TERMS];
    initialize_affine_histories(
        nominal_scalar, residual_scalar, nominal_history, residual_history);

    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        struct affine_value nominal;
        struct affine_value residual;
        // Build yhat_pb(row)=b_pb(row)+a_pb(row)^T*gamma_C.
        build_nominal_affine(axis, row, &nominal);
        // Build ehat_pb(row)=b_e(row)+a_e(row)^T*gamma_C.
        build_residual_affine(
            learner, nominal_history, &nominal, residual_history, &residual);

        // Build y_d(row)-b_h(row), where b_h=b_pb+b_e.
        rhs[row] = desired[row] - nominal.offset - residual.offset;
        int column;
        // Build A_h(row,:)=a_pb(row)^T+a_e(row)^T.
        for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
            matrix[row][column] = nominal.coefficient[column]
                + residual.coefficient[column];
        // yhat_pb(row) becomes a past nominal prediction at row+1.
        shift_affine_history(nominal_history,
                             HYBRID_NOMINAL_TERMS - 1, &nominal);
        // ehat_pb(row) becomes a past predicted residual at row+1.
        shift_affine_history(residual_history,
                             HYBRID_RESIDUAL_TERMS, &residual);
    }
}

/** Prepare delayed history, construct y_h=A_h*gamma_C+b_h, and solve. */
int
fbf_hybrid_solve_coefficients(struct fbf_controller *controller,
                              int axis_index, double start_time,
                              double desired[], double gamma[])
{
    struct fbf_axis *axis = &controller->axis[axis_index];
    struct hybrid_learner *learner = &axis->hybrid;
    // Signal the caller to use standard FBF until enough samples train w.
    if (learner->training_samples < learner->warmup_samples)
        return -1;

    // Align numeric histories with the first sample of this preview window.
    double nominal_scalar[HYBRID_NOMINAL_TERMS - 1];
    double residual_scalar[HYBRID_RESIDUAL_TERMS];
    if (prepare_hybrid_history(controller, axis_index, start_time,
                               nominal_scalar, residual_scalar))
        return -1;

    // Construct A_h and rhs=y_d-b_h for the hybrid least-squares problem.
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS];
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS];
    double rhs[FBF_PREVIEW_SAMPLES];
    build_hybrid_preview_system(
        axis, desired, nominal_scalar, residual_scalar, q, rhs);

    // Solve gamma_C=argmin ||A_h*gamma_C-(y_d-b_h)||_2 by QR.
    if (fbf_factorize_preview_matrix_qr(q, r))
        return -1;
    fbf_solve_qr(q, r, rhs, gamma);
    learner->hybrid_solves++;
    return 0;
}
