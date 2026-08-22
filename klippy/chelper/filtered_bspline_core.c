// Standard and shared filtered B-spline mathematics
//
// Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h> // fabs, isfinite, sqrt
#include <string.h> // memmove, memset
#include "filtered_bspline_internal.h"


/****************************************************************
 * B-spline and filtered-basis construction
 ****************************************************************/

/** Evaluate one causal cardinal B-spline. */
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

static double
basis_value(int sample_offset)
{
    return cardinal_bspline(
        FBF_DEGREE, sample_offset / (double)FBF_KNOT_SPACING);
}

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

/** Apply the identified plant impulse response to the local basis. */
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

/** Generate and DC-normalize the finite plant impulse response. */
int
fbf_build_impulse_response(double impulse[], int numerator_count,
                           double numerator[], int denominator_count,
                           double denominator[])
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
                * impulse[sample - order];
        impulse[sample] = value;
    }

    double truncated_dc_gain = 0.;
    for (sample = 0; sample < FBF_IMPULSE_SAMPLES; sample++)
        truncated_dc_gain += impulse[sample];
    if (fabs(truncated_dc_gain) < 1.0e-12)
        return -1;
    for (sample = 0; sample < FBF_IMPULSE_SAMPLES; sample++)
        impulse[sample] /= truncated_dc_gain;
    return 0;
}

/** Split [tilde_Phi_PC tilde_Phi_C] and preserve Phi_C for hybrid use. */
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
            axis->filtered_current_basis[row][column]
                = axis->q[row][column]
                = filtered_local_basis
                    [row][FBF_HISTORY_COEFFICIENTS + column];
    }
}

/**
 * Factorise a preview matrix A=Q*R using modified Gram--Schmidt.
 *
 * On entry, q contains A: filtered_current_basis for standard FBF or A_h for
 * hybrid FBF.  On successful return, the same array has been overwritten with
 * the orthonormal matrix Q, while r contains the upper-triangular matrix R.
 * Return -1 if a column is non-finite or too close to zero to factorise, and
 * return 0 otherwise.
 */
int
fbf_factorize_preview_matrix_qr(
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS])
{
    memset(r, 0, FBF_CURRENT_COEFFICIENTS
           * FBF_CURRENT_COEFFICIENTS * sizeof(double));
    int k, j;
    for (j = 0; j < FBF_CURRENT_COEFFICIENTS; j++) {
        int i;
        for (i = 0; i < j; i++) {
            double r_ij = 0.;
            for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
                r_ij += q[k][i] * q[k][j];
            r[i][j] = r_ij;
            for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
                q[k][j] -= r_ij * q[k][i];
        }
        double r_jj = 0.;
        for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
            r_jj += q[k][j] * q[k][j];
        r_jj = sqrt(r_jj);
        if (!isfinite(r_jj) || r_jj < 1.0e-10)
            return -1;
        r[j][j] = r_jj;
        for (k = 0; k < FBF_PREVIEW_SAMPLES; k++)
            q[k][j] /= r_jj;
    }
    return 0;
}

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
    return fbf_factorize_preview_matrix_qr(axis->q, axis->r);
}

int
fbf_axis_configure_model(struct fbf_axis *axis, int numerator_count,
                         double numerator[], int denominator_count,
                         double denominator[])
{
    memset(axis, 0, sizeof(*axis));
    if (fbf_build_impulse_response(axis->impulse, numerator_count, numerator,
                                   denominator_count, denominator)
        || build_filtered_basis_qr(axis))
        return -1;
    axis->configured = 1;
    return 0;
}


/****************************************************************
 * Window optimization and command reconstruction
 ****************************************************************/

void
fbf_solve_qr(
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS],
    double rhs[FBF_PREVIEW_SAMPLES], double gamma[])
{
    int column, row;
    for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++) {
        gamma[column] = 0.;
        for (row = 0; row < FBF_PREVIEW_SAMPLES; row++)
            gamma[column] += q[row][column] * rhs[row];
    }
    for (column = FBF_CURRENT_COEFFICIENTS - 1; column >= 0; column--) {
        int next;
        for (next = column + 1; next < FBF_CURRENT_COEFFICIENTS; next++)
            gamma[column] -= r[column][next] * gamma[next];
        gamma[column] /= r[column][column];
    }
}

void
fbf_solve_standard_coefficients(struct fbf_axis *axis, double desired[],
                                double gamma[])
{
    double rhs[FBF_PREVIEW_SAMPLES];
    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        double past_output = 0.;
        int history_index;
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
            past_output += axis->history[history_index]
                * axis->filtered_history_basis[row][history_index];
        rhs[row] = desired[row] - past_output;
    }
    fbf_solve_qr(axis->q, axis->r, rhs, gamma);
}

void
fbf_calculate_nominal_prediction(struct fbf_axis *axis, double gamma[],
                                 double nominal[])
{
    int row;
    for (row = 0; row < FBF_PREVIEW_SAMPLES; row++) {
        double value = 0.;
        int history_index;
        for (history_index = 0;
             history_index < FBF_HISTORY_COEFFICIENTS; history_index++)
            value += axis->history[history_index]
                * axis->filtered_history_basis[row][history_index];
        int column;
        for (column = 0; column < FBF_CURRENT_COEFFICIENTS; column++)
            value += gamma[column]
                * axis->filtered_current_basis[row][column];
        nominal[row] = value;
    }
}

void
fbf_reconstruct_command(struct fbf_axis *axis, double gamma[],
                        double command[])
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

void
fbf_commit_coefficients(struct fbf_axis *axis, double gamma[])
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
