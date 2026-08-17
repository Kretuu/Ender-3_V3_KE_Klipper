#ifndef __FILTERED_BSPLINE_INTERNAL_H
#define __FILTERED_BSPLINE_INTERNAL_H

#include <stddef.h> // offsetof used by list helpers
#include "list.h" // list_head, list_node

// Fixed dissertation controller design.
#define FBF_SAMPLE_TIME 0.001
#define FBF_DEGREE 5
#define FBF_KNOT_SPACING 10
#define FBF_BATCH_SAMPLES 70
#define FBF_PREVIEW_SAMPLES 140
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

// phi(k) = [1, y_pb(k-q+1...k), e_pb(k-p...k-1)]^T.
#define HYBRID_NOMINAL_TERMS 4
#define HYBRID_RESIDUAL_TERMS 50
#define HYBRID_FEATURES \
    (1 + HYBRID_NOMINAL_TERMS + HYBRID_RESIDUAL_TERMS)
#define HYBRID_MAX_OBSERVATION_GAP 0.020

#define FBF_MODE_DISABLED 0
#define FBF_MODE_STANDARD 1
#define FBF_MODE_HYBRID 2

struct hybrid_learner {
    double regularization;
    int warmup_samples;
    int training_samples, measurement_errors, solve_fallbacks, hybrid_solves;
    int prediction_gap_samples, maximum_prediction_gap_samples;
    double weights[HYBRID_FEATURES];
    double covariance[HYBRID_FEATURES][HYBRID_FEATURES];
    double nominal_history[HYBRID_NOMINAL_TERMS - 1];
    double residual_history[HYBRID_RESIDUAL_TERMS];
    long long latest_sample_index;
    int have_observation;
    double last_observation_time, last_observed_position;
};

struct fbf_axis {
    int configured;
    double impulse[FBF_IMPULSE_SAMPLES];
    double filtered_history_basis
        [FBF_PREVIEW_SAMPLES][FBF_HISTORY_COEFFICIENTS];
    double filtered_current_basis
        [FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS];
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS];
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS];
    double history[FBF_HISTORY_COEFFICIENTS];
    struct hybrid_learner hybrid;
};

struct fbf_batch {
    double start_time;
    double position[FBF_AXIS_COUNT][FBF_BATCH_SAMPLES + 1];
    double nominal[FBF_AXIS_COUNT][FBF_BATCH_SAMPLES];
    struct list_node node;
};

struct fbf_controller {
    int mode, initialized;
    // Fixed print-time origin for k=0 on the hybrid 1 kHz sample grid.  It is
    // set once per continuous controller trajectory and does not advance with
    // individual G-code moves or FBF batches.
    double trajectory_start_time;
    // Absolute print time of the next 70 ms batch; advances after every solve.
    double next_batch_time;
    double hybrid_regularization;
    int hybrid_warmup_samples;
    double position_min[FBF_AXIS_COUNT], position_max[FBF_AXIS_COUNT];
    struct fbf_axis axis[FBF_AXIS_COUNT];
    struct list_head batches;
};

// Standard FBF and shared numerical operations.
int fbf_axis_configure_model(struct fbf_axis *axis, int numerator_count,
                             double numerator[], int denominator_count,
                             double denominator[]);
int fbf_factorize_preview_matrix_qr(
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS]);
void fbf_solve_qr(
    double q[FBF_PREVIEW_SAMPLES][FBF_CURRENT_COEFFICIENTS],
    double r[FBF_CURRENT_COEFFICIENTS][FBF_CURRENT_COEFFICIENTS],
    double rhs[FBF_PREVIEW_SAMPLES], double gamma[]);
void fbf_solve_standard_coefficients(struct fbf_axis *axis, double desired[],
                                     double gamma[]);
void fbf_calculate_nominal_prediction(struct fbf_axis *axis, double gamma[],
                                      double nominal[]);
void fbf_reconstruct_command(struct fbf_axis *axis, double gamma[],
                             double command[]);
void fbf_commit_coefficients(struct fbf_axis *axis, double gamma[]);

// Hybrid residual learning and delayed preview prediction.
void fbf_hybrid_reset(struct fbf_axis *axis, double initial_position,
                      double regularization, int warmup_samples);
int fbf_hybrid_train_residual_sample(struct fbf_axis *axis, double nominal,
                                     double residual,
                                     long long sample_index);
int fbf_hybrid_lookup_nominal_prediction(struct fbf_controller *controller,
                                         int axis_index, double print_time,
                                         double *nominal);
int fbf_hybrid_solve_coefficients(struct fbf_controller *controller,
                                  int axis_index, double start_time,
                                  double desired[], double gamma[]);

#endif // __FILTERED_BSPLINE_INTERNAL_H
