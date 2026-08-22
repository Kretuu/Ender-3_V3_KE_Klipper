# Limited-preview filtered B-spline feedforward controller
#
# Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import collections
import logging
import threading
import chelper


# Fixed dissertation parameters.  The C implementation uses the same values.
SAMPLE_TIME = 0.001
B_SPLINE_DEGREE = 5
KNOT_SPACING_SAMPLES = 10
BATCH_SAMPLES = 70
PREVIEW_SAMPLES = 140
IMPULSE_RESPONSE_SAMPLES = 20
PREVIEW_TIME = SAMPLE_TIME * PREVIEW_SAMPLES
HYBRID_NOMINAL_TERMS = 4
HYBRID_RESIDUAL_TERMS = 50
HYBRID_FEATURES = 1 + HYBRID_NOMINAL_TERMS + HYBRID_RESIDUAL_TERMS
HYBRID_WORK_QUEUE_LIMIT = 4096
HYBRID_SNAPSHOT_MAX_AGE = 0.5
HYBRID_SNAPSHOT_INTERVAL = 0.050
HYBRID_SNAPSHOT_ADOPTION_INTERVAL = 0.200
EXPERIMENT_TIME_INDEX = 0
EXPERIMENT_MOTOR_POSITION_INDEX = 12
EXPERIMENT_OBSERVED_POSITION_INDEX = 14
EXPERIMENT_OBSERVER_VALID_INDEX = 16
MODE_VALUES = {"standard": 1, "hybrid": 2}
HybridSnapshot = collections.namedtuple("HybridSnapshot", (
    "generation", "latest_sample_index", "history_samples",
    "training_samples", "measurement_errors", "weights",
    "nominal_history", "residual_history"))


class FilteredBSpline:
    """Connect the nominal filtered B-spline controller to Klipper motion."""

    def __init__(self, config):
        self.printer = config.get_printer()
        self.initial_enabled = config.getboolean("enabled", False)
        self.initial_mode = config.getchoice(
            "mode", {name: name for name in MODE_VALUES}, "standard")
        self.hybrid_regularization = config.getfloat(
            "hybrid_regularization", 0.01, above=0.)
        self.hybrid_warmup_seconds = config.getfloat(
            "hybrid_warmup_seconds", 5.5, minval=0.05)
        self.hybrid_warmup_samples = int(round(
            self.hybrid_warmup_seconds / SAMPLE_TIME))
        self.models = {
            axis: self._read_axis_model(config, axis) for axis in ("x", "y")
        }
        self.position_limits = {
            axis: self._read_axis_limits(config, axis) for axis in ("x", "y")
        }
        self.ffi_main, self.ffi_lib = chelper.get_ffi()
        controller = self.ffi_lib.filtered_bspline_controller_alloc()
        if controller == self.ffi_main.NULL:
            raise config.error(
                "Unable to allocate filtered B-spline controller")
        learner = self.ffi_lib.filtered_bspline_learner_alloc()
        if learner == self.ffi_main.NULL:
            self.ffi_lib.filtered_bspline_controller_free(controller)
            raise config.error("Unable to allocate hybrid FBF learner")
        self.controller = self.ffi_main.gc(
            controller, self.ffi_lib.filtered_bspline_controller_free)
        self.hybrid_learner = self.ffi_main.gc(
            learner, self.ffi_lib.filtered_bspline_learner_free)
        self._configure_models(config)
        self._configure_position_limits(config)
        if self.ffi_lib.filtered_bspline_configure_hybrid(
                self.controller, self.hybrid_regularization,
                self.hybrid_warmup_samples):
            raise config.error("Invalid hybrid FBF configuration")
        if self.ffi_lib.filtered_bspline_learner_configure(
                self.hybrid_learner, self.hybrid_regularization,
                self.hybrid_warmup_samples):
            raise config.error("Invalid hybrid FBF learner configuration")

        self.toolhead = None
        self.mcu = None
        self.enabled = False
        self.mode = self.initial_mode
        self.observer_client = None
        self.hybrid_observation_errors = 0
        self.hybrid_worker_drops = 0
        self.hybrid_worker_queue_max = 0
        self.hybrid_work = collections.deque()
        self.hybrid_work_condition = threading.Condition()
        self.hybrid_worker = None
        self.hybrid_worker_running = False
        # Only the learner changes state when weights are frozen.  The
        # observer and hybrid motion controller remain active.
        self.hybrid_learning_state = "stopped"
        self.hybrid_generation = 1
        self.hybrid_trajectory = None
        self.hybrid_trajectory_end_time = None
        self.hybrid_learning_cutoff = None
        self.hybrid_cutoff_reached = set()
        self.hybrid_snapshots = {b'x': None, b'y': None}
        self.applied_hybrid_snapshots = {b'x': None, b'y': None}
        self.hybrid_snapshot_adoptions = {b'x': 0, b'y': 0}
        self.last_hybrid_snapshot_adoption = {b'x': None, b'y': None}
        self.stepper_kinematics = []
        self.original_stepper_kinematics = []
        self.printer.register_event_handler("klippy:connect", self._connect)
        self.printer.register_event_handler(
            "toolhead:set_position", self._handle_set_position)
        self.printer.register_event_handler(
            "klippy:disconnect", self._handle_disconnect)

        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SET_FILTERED_BSPLINE", self.cmd_SET_FILTERED_BSPLINE,
            desc=self.cmd_SET_FILTERED_BSPLINE_help)
        gcode.register_command(
            "FILTERED_BSPLINE_STATUS", self.cmd_FILTERED_BSPLINE_STATUS,
            desc=self.cmd_FILTERED_BSPLINE_STATUS_help)
        gcode.register_command(
            "WAIT_FILTERED_BSPLINE_READY",
            self.cmd_WAIT_FILTERED_BSPLINE_READY,
            desc=self.cmd_WAIT_FILTERED_BSPLINE_READY_help)
        gcode.register_command(
            "MARK_FILTERED_BSPLINE_LEARNING_END",
            self.cmd_MARK_FILTERED_BSPLINE_LEARNING_END,
            desc=self.cmd_MARK_FILTERED_BSPLINE_LEARNING_END_help)
        gcode.register_command(
            "FREEZE_FILTERED_BSPLINE", self.cmd_FREEZE_FILTERED_BSPLINE,
            desc=self.cmd_FREEZE_FILTERED_BSPLINE_help)

    @staticmethod
    def _read_axis_model(config, axis):
        """Read one discrete transfer function in z^-1 coefficient order."""
        return (
            config.getfloatlist("model_%s_numerator" % (axis,)),
            config.getfloatlist("model_%s_denominator" % (axis,)),
        )

    def _configure_models(self, config):
        """Build the truncated filtered-basis matrices once at startup."""
        for axis, (numerator, denominator) in self.models.items():
            axis_byte = axis.encode()
            result = self.ffi_lib.filtered_bspline_configure_axis(
                self.controller, axis_byte, len(numerator), numerator,
                len(denominator), denominator)
            configure_learner = (
                self.ffi_lib.filtered_bspline_learner_configure_axis)
            learner_result = configure_learner(
                self.hybrid_learner, axis_byte, len(numerator),
                numerator, len(denominator), denominator)
            if result or learner_result:
                raise config.error(
                    "Invalid or rank-deficient %s-axis filtered B-spline model"
                    % (axis.upper(),))

    @staticmethod
    def _read_axis_limits(config, axis):
        """Read the existing Klipper travel range for one physical axis."""
        stepper_config = config.getsection("stepper_%s" % (axis,))
        position_min = stepper_config.getfloat("position_min", 0.)
        position_max = stepper_config.getfloat(
            "position_max", above=position_min)
        return position_min, position_max

    def _configure_position_limits(self, config):
        """Reject compensated samples outside the configured travel range."""
        for axis, limits in self.position_limits.items():
            result = self.ffi_lib.filtered_bspline_set_axis_limits(
                self.controller, axis.encode(), limits[0], limits[1])
            if result:
                raise config.error(
                    "Invalid %s-axis limits for filtered B-spline control"
                    % (axis.upper(),))

    def _connect(self):
        """Wrap the existing XY stepper kinematics after Klipper connects."""
        if self.printer.lookup_object("input_shaper", None) is not None:
            raise self.printer.config_error(
                "filtered_bspline and input_shaper cannot be enabled together")

        self.toolhead = self.printer.lookup_object("toolhead")
        self.mcu = self.printer.lookup_object("mcu")
        for stepper in self.toolhead.get_kinematics().get_steppers():
            wrapper = self.ffi_main.gc(
                self.ffi_lib.filtered_bspline_stepper_alloc(self.controller),
                self.ffi_lib.free)
            if wrapper == self.ffi_main.NULL:
                raise self.printer.config_error(
                    "Unable to allocate filtered B-spline stepper kinematics")
            original = stepper.set_stepper_kinematics(wrapper)
            if self.ffi_lib.filtered_bspline_stepper_set_sk(
                    wrapper, original) < 0:
                stepper.set_stepper_kinematics(original)
                continue
            self.stepper_kinematics.append(wrapper)
            # The C wrapper retains only the original kinematics' raw pointer.
            # Keep its ffi.gc owner alive so it is not freed after replacement.
            self.original_stepper_kinematics.append(original)

        if not self.stepper_kinematics:
            raise self.printer.config_error(
                "filtered_bspline did not find an X or Y stepper")
        self.toolhead.register_motion_prepare_callback(self._prepare_motion)
        if self.initial_enabled:
            self._set_state(True, self.initial_mode)

    def _start_observer_client(self):
        """Subscribe to the one observer calculation shared with Motan."""
        if self.observer_client is not None:
            return
        trinkey = self.printer.lookup_object("trinkey_accel", None)
        if trinkey is None:
            raise self.printer.command_error(
                "Hybrid FBF requires a configured [trinkey_accel]")
        self._start_hybrid_worker()
        try:
            self.observer_client = trinkey.start_observer_client(
                self._handle_observer_message)
        except Exception:
            self._stop_hybrid_worker()
            raise

    def _stop_observer_client(self):
        if self.observer_client is None:
            self._stop_hybrid_worker()
            return
        self.observer_client.finalize()
        self.observer_client = None
        self._stop_hybrid_worker()

    def _check_hybrid_observer(self):
        """Fail learning when its shared accelerometer stream has stopped."""
        if (self.observer_client is not None
                and self.observer_client.is_closed()):
            self._fail_hybrid_learning()
            raise self.printer.command_error(
                "Hybrid FBF observer stream stopped")

    def _fail_hybrid_learning(self):
        """Close an unusable training run without touching controller maths."""
        with self.hybrid_work_condition:
            self.hybrid_worker_running = False
            self.hybrid_learning_state = "failed"
            self.hybrid_work.clear()
            self.hybrid_work_condition.notify_all()
        if self.observer_client is not None:
            self.observer_client.finalize()

    def _handle_observer_message(self, message):
        """Immediately hand one completed observer batch to the RLS worker."""
        # A marked endpoint defines the precise training interval.  Later
        # observations remain available to Motan, but are not sent to RLS.
        if self.hybrid_learning_state != "learning":
            return
        trajectory = self.hybrid_trajectory
        if trajectory is None:
            return
        generation, trajectory_start_time = trajectory
        work = []
        cutoff_reached = set()
        params = message.get("params", {})
        for sensor, axis in (("toolhead", "x"), ("base", "y")):
            axis_byte = axis.encode()
            for row in params.get(sensor, ()):
                if (len(row) <= EXPERIMENT_OBSERVER_VALID_INDEX
                        or not row[EXPERIMENT_OBSERVER_VALID_INDEX]):
                    continue
                print_time = row[EXPERIMENT_TIME_INDEX]
                cutoff = self.hybrid_learning_cutoff
                if cutoff is not None and print_time > cutoff:
                    cutoff_reached.add(axis_byte)
                    continue
                work.append((
                    axis_byte, generation, trajectory_start_time, print_time,
                    row[EXPERIMENT_MOTOR_POSITION_INDEX],
                    row[EXPERIMENT_OBSERVED_POSITION_INDEX]))
        self._enqueue_observer_work(work)
        if cutoff_reached:
            with self.hybrid_work_condition:
                self.hybrid_cutoff_reached.update(cutoff_reached)
                self.hybrid_work_condition.notify_all()

    def _enqueue_observer_work(self, work):
        if not work:
            return
        with self.hybrid_work_condition:
            if (not self.hybrid_worker_running
                    or self.hybrid_learning_state != "learning"):
                return
            overflow = (len(self.hybrid_work) + len(work)
                        - HYBRID_WORK_QUEUE_LIMIT)
            if overflow > 0:
                drop_count = min(overflow, len(self.hybrid_work))
                for unused in range(drop_count):
                    self.hybrid_work.popleft()
                overflow -= drop_count
                if overflow > 0:
                    work = work[overflow:]
                    drop_count += overflow
                self.hybrid_worker_drops += drop_count
            self.hybrid_work.extend(work)
            if len(self.hybrid_work) > self.hybrid_worker_queue_max:
                self.hybrid_worker_queue_max = len(self.hybrid_work)
            self.hybrid_work_condition.notify()

    def _start_hybrid_worker(self):
        if self.hybrid_worker is not None:
            if self.hybrid_worker.is_alive():
                return
            self.hybrid_worker = None
        with self.hybrid_work_condition:
            self.hybrid_work.clear()
            self.hybrid_worker_running = True
            self.hybrid_learning_state = "learning"
        self.hybrid_worker = threading.Thread(
            target=self._hybrid_worker_loop,
            name="filtered-bspline-learner")
        self.hybrid_worker.daemon = True
        self.hybrid_worker.start()

    def _stop_hybrid_worker(self):
        worker = self.hybrid_worker
        if worker is None:
            self.hybrid_learning_state = "stopped"
            return
        with self.hybrid_work_condition:
            self.hybrid_worker_running = False
            self.hybrid_learning_state = "stopped"
            self.hybrid_work.clear()
            self.hybrid_work_condition.notify_all()
        worker.join(1.0)
        if worker.is_alive():
            logging.warning("Filtered B-spline learner did not stop promptly")
        else:
            self.hybrid_worker = None

    def _hybrid_worker_loop(self):
        """Train private RLS state and publish complete immutable snapshots."""
        buffers = {}
        last_publication = {b'x': None, b'y': None}
        publication_generation = {b'x': None, b'y': None}
        for axis in (b'x', b'y'):
            buffers[axis] = (
                self.ffi_main.new("int *"), self.ffi_main.new("int *"),
                self.ffi_main.new("int *"),
                self.ffi_main.new("long long *"),
                self.ffi_main.new("double[]", HYBRID_FEATURES),
                self.ffi_main.new("double[]", HYBRID_NOMINAL_TERMS - 1),
                self.ffi_main.new("double[]", HYBRID_RESIDUAL_TERMS))
        while True:
            finalize_learning = False
            with self.hybrid_work_condition:
                while (self.hybrid_worker_running
                       and not self.hybrid_work
                       and self.hybrid_learning_state != "freezing"):
                    self.hybrid_work_condition.wait()
                if not self.hybrid_worker_running:
                    return
                if self.hybrid_work:
                    work = self.hybrid_work.popleft()
                else:
                    # No new observations can enter while FREEZING.  Reaching
                    # an empty queue therefore means every observation before
                    # the cutoff has been processed.
                    finalize_learning = True
            if finalize_learning:
                for axis in (b'x', b'y'):
                    generation = publication_generation[axis]
                    if generation is None:
                        continue
                    try:
                        self._publish_hybrid_snapshot(
                            axis, generation, buffers[axis])
                    except Exception:
                        self.hybrid_observation_errors += 1
                        logging.exception(
                            "Filtered B-spline final snapshot failed")
                with self.hybrid_work_condition:
                    if self.hybrid_learning_state == "freezing":
                        self.hybrid_learning_state = "frozen"
                    self.hybrid_work_condition.notify_all()
                continue
            (axis, generation, trajectory_start_time, print_time,
             motor_position, observed_position) = work
            try:
                result = self.ffi_lib.filtered_bspline_learner_add_observation(
                    self.hybrid_learner, axis, generation,
                    trajectory_start_time, print_time, motor_position,
                    observed_position)
                if result < 0:
                    self.hybrid_observation_errors += 1
                    continue
                if publication_generation[axis] != generation:
                    publication_generation[axis] = generation
                    last_publication[axis] = None
                last_time = last_publication[axis]
                if (last_time is None
                        or print_time - last_time
                           >= HYBRID_SNAPSHOT_INTERVAL):
                    self._publish_hybrid_snapshot(
                        axis, generation, buffers[axis])
                    last_publication[axis] = print_time
            except Exception:
                self.hybrid_observation_errors += 1
                logging.exception("Filtered B-spline learner update failed")

    def _publish_hybrid_snapshot(self, axis, generation, buffers):
        """Replace one snapshot after all of its fields are copied."""
        (history_samples, training_samples, measurement_errors,
         latest_sample_index, weights, nominal_history,
         residual_history) = buffers
        result = self.ffi_lib.filtered_bspline_learner_get_snapshot(
            self.hybrid_learner, axis, history_samples,
            training_samples, measurement_errors, latest_sample_index,
            weights, nominal_history, residual_history)
        if result or latest_sample_index[0] < 0:
            return
        self.hybrid_snapshots[axis] = HybridSnapshot(
            generation, latest_sample_index[0], history_samples[0],
            training_samples[0], measurement_errors[0], tuple(weights),
            tuple(nominal_history), tuple(residual_history))

    def _snapshot_age(self, snapshot, eventtime):
        trajectory = self.hybrid_trajectory
        if (snapshot is None or trajectory is None
                or snapshot.generation != trajectory[0]):
            return None
        sample_time = (trajectory[1]
                       + snapshot.latest_sample_index * SAMPLE_TIME)
        return max(0., self.mcu.estimated_print_time(eventtime) - sample_time)

    def _apply_hybrid_snapshots(self, eventtime, force=False):
        """Adopt publications without discarding older valid weights.

        Normal motion planning only adopts recent snapshots at a bounded rate.
        Freeze uses force=True once, after the worker is idle, to install the
        exact final weights even if the last training sample is now old.
        """
        worker = self.hybrid_worker
        if worker is None or not worker.is_alive():
            for axis in (b'x', b'y'):
                if self.applied_hybrid_snapshots[axis] is not None:
                    self.ffi_lib.filtered_bspline_clear_hybrid_snapshot(
                        self.controller, axis)
                    self.applied_hybrid_snapshots[axis] = None
                    self.last_hybrid_snapshot_adoption[axis] = None
            return
        for axis in (b'x', b'y'):
            snapshot = self.hybrid_snapshots[axis]
            snapshot_age = self._snapshot_age(snapshot, eventtime)
            if (snapshot_age is None
                    or (not force
                        and snapshot_age > HYBRID_SNAPSHOT_MAX_AGE)):
                continue
            if snapshot is self.applied_hybrid_snapshots[axis]:
                continue
            last_adoption = self.last_hybrid_snapshot_adoption[axis]
            if (not force and last_adoption is not None
                    and eventtime - last_adoption
                        < HYBRID_SNAPSHOT_ADOPTION_INTERVAL):
                continue
            result = self.ffi_lib.filtered_bspline_apply_hybrid_snapshot(
                self.controller, axis, snapshot.latest_sample_index,
                snapshot.history_samples, snapshot.training_samples,
                snapshot.measurement_errors, snapshot.weights,
                snapshot.nominal_history, snapshot.residual_history)
            if result:
                self.hybrid_observation_errors += 1
                continue
            self.applied_hybrid_snapshots[axis] = snapshot
            self.last_hybrid_snapshot_adoption[axis] = eventtime
            self.hybrid_snapshot_adoptions[axis] += 1

    def _prepare_motion(self, start_time, end_time, is_final):
        """Solve every 70-sample batch with a complete 140-sample preview."""
        eventtime = self.printer.get_reactor().monotonic()
        self._apply_hybrid_snapshots(eventtime)
        result = self.ffi_lib.filtered_bspline_prepare(
            self.controller, self.toolhead.get_trapq(), start_time,
            end_time, is_final)
        if result == -2:
            raise self.printer.command_error(
                "Filtered B-spline motion exceeds an axis travel limit")
        if result:
            raise self.printer.command_error(
                "Unable to prepare filtered B-spline motion")
        initialized = self.ffi_main.new("int *")
        trajectory_start_time = self.ffi_main.new("double *")
        self.ffi_lib.filtered_bspline_get_trajectory(
            self.controller, initialized, trajectory_start_time)
        if initialized[0]:
            self.hybrid_trajectory = (
                self.hybrid_generation, trajectory_start_time[0])
            self.hybrid_trajectory_end_time = end_time

    def _handle_set_position(self):
        """Discard coefficients when homing or G92 changes the coordinates."""
        self.ffi_lib.filtered_bspline_controller_reset(self.controller)
        self._reset_hybrid_trajectory()

    def _reset_hybrid_trajectory(self):
        """Invalidate publications and resume learning on the new time grid."""
        self.hybrid_generation += 1
        self.hybrid_trajectory = None
        self.hybrid_trajectory_end_time = None
        self.hybrid_learning_cutoff = None
        self.hybrid_cutoff_reached.clear()
        self.hybrid_snapshots = {b'x': None, b'y': None}
        self.applied_hybrid_snapshots = {b'x': None, b'y': None}
        self.last_hybrid_snapshot_adoption = {b'x': None, b'y': None}
        with self.hybrid_work_condition:
            self.hybrid_work.clear()
            if self.hybrid_worker_running:
                # Homing or G92 invalidates the trajectory to which frozen
                # weights belong, so a new trajectory must learn afresh.
                self.hybrid_learning_state = "learning"
                self.hybrid_work_condition.notify_all()

    def _handle_disconnect(self):
        self._stop_observer_client()

    def _set_state(self, enabled, mode):
        """Change controller mode only after all old steps are flushed."""
        if enabled == self.enabled and mode == self.mode:
            return
        if not enabled and not self.enabled:
            self.mode = mode
            return
        old_delay = PREVIEW_TIME if self.enabled else 0.
        new_delay = PREVIEW_TIME if enabled else 0.
        self.toolhead.flush_step_generation()
        if enabled and mode == "hybrid":
            self._start_observer_client()
        self.toolhead.note_step_generation_scan_time(
            new_delay, old_delay=old_delay)
        c_mode = MODE_VALUES[mode] if enabled else 0
        if self.ffi_lib.filtered_bspline_set_mode(self.controller, c_mode):
            raise self.printer.command_error(
                "Unable to change filtered B-spline mode")
        self._reset_hybrid_trajectory()
        for wrapper in self.stepper_kinematics:
            # Include pre-action and settling outside nominal axis moves.
            self.ffi_lib.filtered_bspline_stepper_set_generation_window(
                wrapper, new_delay)
        self.enabled = enabled
        self.mode = mode
        # Keep the completed run's diagnostics visible after ENABLE=0.  A new
        # enabled run starts a fresh count, matching the C learner reset at
        # trajectory initialization.
        if enabled:
            self.hybrid_observation_errors = 0
            self.hybrid_worker_drops = 0
            self.hybrid_worker_queue_max = 0
            self.hybrid_snapshot_adoptions = {b'x': 0, b'y': 0}
        if not enabled or mode != "hybrid":
            self._stop_observer_client()

    cmd_SET_FILTERED_BSPLINE_help = (
        "Select and enable standard or hybrid filtered B-spline feedforward")

    def cmd_SET_FILTERED_BSPLINE(self, gcmd):
        """Handle ENABLE=0|1 MODE=STANDARD|HYBRID."""
        enabled = bool(gcmd.get_int(
            "ENABLE", int(self.enabled), minval=0, maxval=1))
        mode = gcmd.get("MODE", self.mode).lower()
        if mode not in MODE_VALUES:
            raise gcmd.error("MODE must be STANDARD or HYBRID")
        self._set_state(enabled, mode)
        gcmd.respond_info(
            "filtered_bspline enabled:%d mode:%s sample_time:%.4f "
            "preview_time:%.3f"
            % (self.enabled, self.mode, SAMPLE_TIME, PREVIEW_TIME))

    def _get_hybrid_axis_status(self, axis):
        eventtime = self.printer.get_reactor().monotonic()
        history_samples = self.ffi_main.new("int *")
        training_samples = self.ffi_main.new("int *")
        active = self.ffi_main.new("int *")
        measurement_errors = self.ffi_main.new("int *")
        solve_fallbacks = self.ffi_main.new("int *")
        hybrid_solves = self.ffi_main.new("int *")
        prediction_gap_samples = self.ffi_main.new("int *")
        maximum_prediction_gap_samples = self.ffi_main.new("int *")
        history_replay_samples = self.ffi_main.new("int *")
        maximum_history_replay_samples = self.ffi_main.new("int *")
        weight_norm = self.ffi_main.new("double *")
        axis_byte = axis.encode()
        self.ffi_lib.filtered_bspline_get_hybrid_status(
            self.controller, axis_byte, history_samples, training_samples,
            active, measurement_errors, solve_fallbacks, hybrid_solves,
            prediction_gap_samples, maximum_prediction_gap_samples,
            history_replay_samples, maximum_history_replay_samples,
            weight_norm)
        applied_history_samples = history_samples[0]
        applied_training_samples = training_samples[0]
        snapshot = self.hybrid_snapshots[axis_byte]
        snapshot_age = None
        if (snapshot is not None
                and snapshot.generation == self.hybrid_generation):
            snapshot_age = self._snapshot_age(snapshot, eventtime)
            history_samples[0] = snapshot.history_samples
            training_samples[0] = snapshot.training_samples
            measurement_errors[0] = snapshot.measurement_errors
        applied_snapshot = self.applied_hybrid_snapshots[axis_byte]
        applied_snapshot_age = self._snapshot_age(applied_snapshot, eventtime)
        return collections.OrderedDict([
            ("history_samples", history_samples[0]),
            ("applied_history_samples", applied_history_samples),
            ("training_samples", training_samples[0]),
            ("applied_training_samples", applied_training_samples),
            ("active", bool(active[0])),
            ("measurement_errors", measurement_errors[0]),
            ("solve_fallbacks", solve_fallbacks[0]),
            ("hybrid_solves", hybrid_solves[0]),
            ("prediction_gap_samples", prediction_gap_samples[0]),
            ("maximum_prediction_gap_samples",
             maximum_prediction_gap_samples[0]),
            ("history_replay_samples", history_replay_samples[0]),
            ("maximum_history_replay_samples",
             maximum_history_replay_samples[0]),
            ("weight_norm", weight_norm[0]),
            ("snapshot_fresh", snapshot_age is not None
             and snapshot_age <= HYBRID_SNAPSHOT_MAX_AGE),
            ("snapshot_age", -1. if snapshot_age is None else snapshot_age),
            ("applied_snapshot_age", -1. if applied_snapshot_age is None
             else applied_snapshot_age),
            ("snapshot_adoptions",
             self.hybrid_snapshot_adoptions[axis_byte]),
        ])

    cmd_FILTERED_BSPLINE_STATUS_help = (
        "Report standard/hybrid filtered B-spline controller state")

    def cmd_FILTERED_BSPLINE_STATUS(self, gcmd):
        lines = ["filtered_bspline enabled:%d mode:%s learning:%s" % (
            self.enabled, self.mode, self.hybrid_learning_state)]
        for axis in ("x", "y"):
            status = self._get_hybrid_axis_status(axis)
            lines.append(
                "%s history:%d applied_history:%d "
                "rls_updates:%d applied_rls_updates:%d active:%d "
                "hybrid_solves:%d "
                "snapshot_fresh:%d snapshot_age:%.3f "
                "applied_snapshot_age:%.3f snapshot_adoptions:%d "
                "gap_samples:%d max_gap_samples:%d "
                "replay_samples:%d max_replay_samples:%d "
                "measurement_errors:%d solve_fallbacks:%d "
                "weight_norm:%.6g" % (
                    axis, status["history_samples"],
                    status["applied_history_samples"],
                    status["training_samples"],
                    status["applied_training_samples"], status["active"],
                    status["hybrid_solves"],
                    status["snapshot_fresh"], status["snapshot_age"],
                    status["applied_snapshot_age"],
                    status["snapshot_adoptions"],
                    status["prediction_gap_samples"],
                    status["maximum_prediction_gap_samples"],
                    status["history_replay_samples"],
                    status["maximum_history_replay_samples"],
                    status["measurement_errors"],
                    status["solve_fallbacks"], status["weight_norm"]))
        lines.append("python_observation_errors:%d" % (
            self.hybrid_observation_errors,))
        lines.append(
            "worker_queue:%d worker_queue_max:%d worker_drops:%d "
            "worker_alive:%d" % (
            len(self.hybrid_work), self.hybrid_worker_queue_max,
            self.hybrid_worker_drops,
            self.hybrid_worker is not None and self.hybrid_worker.is_alive()))
        gcmd.respond_info("\n".join(lines))

    cmd_WAIT_FILTERED_BSPLINE_READY_help = (
        "Wait until both hybrid residual learners complete warm-up")

    def cmd_WAIT_FILTERED_BSPLINE_READY(self, gcmd):
        if not self.enabled or self.mode != "hybrid":
            raise gcmd.error(
                "WAIT_FILTERED_BSPLINE_READY requires enabled hybrid mode")
        timeout = gcmd.get_float("TIMEOUT", 2., above=0.)
        reactor = self.printer.get_reactor()
        deadline = reactor.monotonic() + timeout
        while True:
            self._check_hybrid_observer()
            eventtime = reactor.monotonic()
            self._apply_hybrid_snapshots(eventtime)
            x_status = self._get_hybrid_axis_status("x")
            y_status = self._get_hybrid_axis_status("y")
            if (x_status["active"] and y_status["active"]
                    and x_status["snapshot_fresh"]
                    and y_status["snapshot_fresh"]):
                gcmd.respond_info(
                    "Hybrid FBF ready: x_history:%d y_history:%d "
                    "x_rls_updates:%d y_rls_updates:%d"
                    % (x_status["history_samples"],
                       y_status["history_samples"],
                       x_status["training_samples"],
                       y_status["training_samples"]))
                return
            if eventtime >= deadline:
                self._fail_hybrid_learning()
                raise gcmd.error(
                    "Hybrid FBF warm-up timed out: x_history:%d "
                    "y_history:%d required:%d x_fresh:%d y_fresh:%d"
                    % (x_status["history_samples"],
                       y_status["history_samples"],
                       self.hybrid_warmup_samples,
                       x_status["snapshot_fresh"],
                       y_status["snapshot_fresh"]))
            reactor.pause(min(deadline, eventtime + .050))

    cmd_MARK_FILTERED_BSPLINE_LEARNING_END_help = (
        "Mark the end of the queued hybrid RLS training motion")

    def cmd_MARK_FILTERED_BSPLINE_LEARNING_END(self, gcmd):
        """Exclude observations after the currently queued training motion."""
        if not self.enabled or self.mode != "hybrid":
            raise gcmd.error(
                "MARK_FILTERED_BSPLINE_LEARNING_END requires hybrid mode")
        if self.hybrid_learning_state != "learning":
            raise gcmd.error("Hybrid FBF learner is not accepting samples")
        self._check_hybrid_observer()

        # Flushing lookahead calls _prepare_motion(), which provides the exact
        # end of the physical trajectory without the step-generation padding.
        self.toolhead.get_last_move_time()
        if self.hybrid_trajectory_end_time is None:
            raise gcmd.error("No hybrid training motion has been queued")
        self.hybrid_learning_cutoff = self.hybrid_trajectory_end_time
        self.hybrid_cutoff_reached.clear()
        gcmd.respond_info(
            "Hybrid FBF learning endpoint marked at print_time %.6f"
            % (self.hybrid_learning_cutoff,))

    cmd_FREEZE_FILTERED_BSPLINE_help = (
        "Drain hybrid RLS work and keep its final learned weights fixed")

    def cmd_FREEZE_FILTERED_BSPLINE(self, gcmd):
        """Stop RLS at a defined cutoff and atomically adopt its final result."""
        if not self.enabled or self.mode != "hybrid":
            raise gcmd.error(
                "FREEZE_FILTERED_BSPLINE requires enabled hybrid mode")
        if self.hybrid_learning_state == "frozen":
            gcmd.respond_info("Hybrid FBF learning is already frozen")
            return
        if self.hybrid_learning_state != "learning":
            raise gcmd.error("Hybrid FBF learner is not accepting samples")

        timeout = gcmd.get_float("TIMEOUT", 5., above=0.)
        reactor = self.printer.get_reactor()
        deadline = reactor.monotonic() + timeout

        # If an endpoint was marked before M400, wait until both observer
        # streams have crossed it.  This admits every delayed warm-up sample
        # while excluding the stationary samples which follow the warm-up.
        while (self.hybrid_learning_cutoff is not None
               and self.hybrid_cutoff_reached != {b'x', b'y'}):
            self._check_hybrid_observer()
            eventtime = reactor.monotonic()
            if eventtime >= deadline:
                self._fail_hybrid_learning()
                raise gcmd.error(
                    "Timed out waiting for observer data to cross the "
                    "hybrid learning endpoint")
            reactor.pause(min(deadline, eventtime + .020))

        with self.hybrid_work_condition:
            self.hybrid_learning_state = "freezing"
            self.hybrid_work_condition.notify_all()

        while self.hybrid_learning_state != "frozen":
            self._check_hybrid_observer()
            eventtime = reactor.monotonic()
            if eventtime >= deadline:
                self._fail_hybrid_learning()
                raise gcmd.error(
                    "Timed out draining hybrid RLS work; learning aborted")
            reactor.pause(min(deadline, eventtime + .020))

        # The final publication may be older than the normal live-adoption
        # limit after the warm-up M400.  It is intentionally adopted once here
        # because no later RLS update can replace it during evaluation.
        self._apply_hybrid_snapshots(reactor.monotonic(), force=True)
        for axis in (b'x', b'y'):
            if (self.hybrid_snapshots[axis]
                    is not self.applied_hybrid_snapshots[axis]):
                self._fail_hybrid_learning()
                raise gcmd.error(
                    "Unable to adopt final hybrid %s-axis weights"
                    % (axis.decode().upper(),))

        x_status = self._get_hybrid_axis_status("x")
        y_status = self._get_hybrid_axis_status("y")
        if not x_status["active"] or not y_status["active"]:
            self._fail_hybrid_learning()
            raise gcmd.error(
                "Hybrid FBF warm-up did not produce usable learned weights")
        gcmd.respond_info(
            "Hybrid FBF learning frozen: x_rls_updates:%d "
            "y_rls_updates:%d"
            % (x_status["applied_training_samples"],
               y_status["applied_training_samples"]))

    def get_status(self, eventtime):
        """Expose the fixed experimental design through Klipper status."""
        return collections.OrderedDict([
            ("enabled", self.enabled),
            ("mode", self.mode),
            ("sample_time", SAMPLE_TIME),
            ("degree", B_SPLINE_DEGREE),
            ("knot_spacing_samples", KNOT_SPACING_SAMPLES),
            ("batch_samples", BATCH_SAMPLES),
            ("preview_samples", PREVIEW_SAMPLES),
            ("impulse_response_samples", IMPULSE_RESPONSE_SAMPLES),
            ("hybrid_nominal_terms", HYBRID_NOMINAL_TERMS),
            ("hybrid_residual_terms", HYBRID_RESIDUAL_TERMS),
            ("hybrid_regularization", self.hybrid_regularization),
            ("hybrid_warmup_seconds", self.hybrid_warmup_seconds),
            ("hybrid_snapshot_max_age", HYBRID_SNAPSHOT_MAX_AGE),
            ("hybrid_snapshot_interval", HYBRID_SNAPSHOT_INTERVAL),
            ("hybrid_snapshot_adoption_interval",
             HYBRID_SNAPSHOT_ADOPTION_INTERVAL),
            ("hybrid_observation_errors", self.hybrid_observation_errors),
            ("hybrid_learning_state", self.hybrid_learning_state),
            ("hybrid_worker_queue", len(self.hybrid_work)),
            ("hybrid_worker_queue_max", self.hybrid_worker_queue_max),
            ("hybrid_worker_drops", self.hybrid_worker_drops),
            ("hybrid_worker_alive", self.hybrid_worker is not None
             and self.hybrid_worker.is_alive()),
            ("hybrid_x", self._get_hybrid_axis_status("x")),
            ("hybrid_y", self._get_hybrid_axis_status("y")),
        ])


def load_config(config):
    return FilteredBSpline(config)
