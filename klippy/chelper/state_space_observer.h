#ifndef STATE_SPACE_OBSERVER_H
#define STATE_SPACE_OBSERVER_H

struct state_space_observer;

struct state_space_observer *state_space_observer_alloc(
    int state_count, double fo[], double gu[], double gv[], double ga[],
    double ho[], double x0_per_unit[]);
void state_space_observer_free(struct state_space_observer *observer);
void state_space_observer_reset(struct state_space_observer *observer);
double state_space_observer_sample(struct state_space_observer *observer,
                                   double input_position,
                                   double input_velocity,
                                   double measured_acceleration);

#endif // STATE_SPACE_OBSERVER_H
