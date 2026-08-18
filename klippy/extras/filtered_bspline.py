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
HYBRID_WORK_QUEUE_LIMIT = 4096
EXPERIMENT_TIME_INDEX = 0
EXPERIMENT_OBSERVED_POSITION_INDEX = 14
EXPERIMENT_OBSERVER_VALID_INDEX = 16
MODE_VALUES = {"standard": 1, "hybrid": 2}


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
        self.controller = self.ffi_main.gc(
            controller, self.ffi_lib.filtered_bspline_controller_free)
        self._configure_models(config)
        self._configure_position_limits(config)
        if self.ffi_lib.filtered_bspline_configure_hybrid(
                self.controller, self.hybrid_regularization,
                self.hybrid_warmup_samples):
            raise config.error("Invalid hybrid FBF configuration")

        self.toolhead = None
        self.enabled = False
        self.mode = self.initial_mode
        self.observer_client = None
        self.hybrid_observation_errors = 0
        self.hybrid_worker_drops = 0
        self.hybrid_worker_queue_max = 0
        self.hybrid_work = collections.deque()
        self.hybrid_work_condition = threading.Condition()
        self.hybrid_controller_lock = threading.Lock()
        self.hybrid_worker = None
        self.hybrid_worker_running = False
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
            result = self.ffi_lib.filtered_bspline_configure_axis(
                self.controller, axis.encode(), len(numerator), numerator,
                len(denominator), denominator)
            if result:
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

    def _handle_observer_message(self, message):
        """Immediately hand one completed observer batch to the RLS worker."""
        work = []
        params = message.get("params", {})
        for sensor, axis in (("toolhead", "x"), ("base", "y")):
            for row in params.get(sensor, ()):
                if (len(row) <= EXPERIMENT_OBSERVER_VALID_INDEX
                        or not row[EXPERIMENT_OBSERVER_VALID_INDEX]):
                    continue
                work.append((
                    axis.encode(), row[EXPERIMENT_TIME_INDEX],
                    row[EXPERIMENT_OBSERVED_POSITION_INDEX]))
        self._enqueue_observer_work(work)

    def _enqueue_observer_work(self, work):
        if not work:
            return
        with self.hybrid_work_condition:
            if not self.hybrid_worker_running:
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
        self.hybrid_worker = threading.Thread(
            target=self._hybrid_worker_loop,
            name="filtered-bspline-learner")
        self.hybrid_worker.daemon = True
        self.hybrid_worker.start()

    def _stop_hybrid_worker(self):
        worker = self.hybrid_worker
        if worker is None:
            return
        with self.hybrid_work_condition:
            self.hybrid_worker_running = False
            self.hybrid_work.clear()
            self.hybrid_work_condition.notify_all()
        worker.join(1.0)
        if worker.is_alive():
            logging.warning("Filtered B-spline learner did not stop promptly")
        else:
            self.hybrid_worker = None

    def _hybrid_worker_loop(self):
        """Train RLS off-reactor while serializing mutable controller access."""
        while True:
            with self.hybrid_work_condition:
                while (self.hybrid_worker_running
                       and not self.hybrid_work):
                    self.hybrid_work_condition.wait()
                if not self.hybrid_worker_running:
                    return
                axis, print_time, observed_position = self.hybrid_work.popleft()
            try:
                # One observation performs at most four 1 kHz updates.  Taking
                # the lock per observation bounds reactor contention while the
                # CFFI call executes on a worker-capable native thread.
                with self.hybrid_controller_lock:
                    result = self.ffi_lib.filtered_bspline_add_observation(
                        self.controller, axis, print_time, observed_position)
                if result < 0:
                    self.hybrid_observation_errors += 1
            except Exception:
                self.hybrid_observation_errors += 1
                logging.exception("Filtered B-spline learner update failed")

    def _prepare_motion(self, start_time, end_time, is_final):
        """Solve every 70-sample batch with a complete 140-sample preview."""
        # The worker never touches TrapQ.  This lock only prevents it from
        # reading controller batches while prepare mutates or discards them.
        with self.hybrid_controller_lock:
            result = self.ffi_lib.filtered_bspline_prepare(
                self.controller, self.toolhead.get_trapq(), start_time,
                end_time, is_final)
        if result == -2:
            raise self.printer.command_error(
                "Filtered B-spline motion exceeds an axis travel limit")
        if result:
            raise self.printer.command_error(
                "Unable to prepare filtered B-spline motion")

    def _handle_set_position(self):
        """Discard coefficients when homing or G92 changes the coordinates."""
        with self.hybrid_controller_lock:
            self.ffi_lib.filtered_bspline_controller_reset(self.controller)
        with self.hybrid_work_condition:
            self.hybrid_work.clear()

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
        with self.hybrid_controller_lock:
            if self.ffi_lib.filtered_bspline_set_mode(
                    self.controller, c_mode):
                raise self.printer.command_error(
                    "Unable to change filtered B-spline mode")
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
        training_samples = self.ffi_main.new("int *")
        active = self.ffi_main.new("int *")
        measurement_errors = self.ffi_main.new("int *")
        solve_fallbacks = self.ffi_main.new("int *")
        hybrid_solves = self.ffi_main.new("int *")
        prediction_gap_samples = self.ffi_main.new("int *")
        maximum_prediction_gap_samples = self.ffi_main.new("int *")
        weight_norm = self.ffi_main.new("double *")
        with self.hybrid_controller_lock:
            self.ffi_lib.filtered_bspline_get_hybrid_status(
                self.controller, axis.encode(), training_samples, active,
                measurement_errors, solve_fallbacks, hybrid_solves,
                prediction_gap_samples, maximum_prediction_gap_samples,
                weight_norm)
        return collections.OrderedDict([
            ("training_samples", training_samples[0]),
            ("active", bool(active[0])),
            ("measurement_errors", measurement_errors[0]),
            ("solve_fallbacks", solve_fallbacks[0]),
            ("hybrid_solves", hybrid_solves[0]),
            ("prediction_gap_samples", prediction_gap_samples[0]),
            ("maximum_prediction_gap_samples",
             maximum_prediction_gap_samples[0]),
            ("weight_norm", weight_norm[0]),
        ])

    cmd_FILTERED_BSPLINE_STATUS_help = (
        "Report standard/hybrid filtered B-spline controller state")

    def cmd_FILTERED_BSPLINE_STATUS(self, gcmd):
        lines = ["filtered_bspline enabled:%d mode:%s" % (
            self.enabled, self.mode)]
        for axis in ("x", "y"):
            status = self._get_hybrid_axis_status(axis)
            lines.append(
                "%s training:%d active:%d hybrid_solves:%d "
                "gap_samples:%d max_gap_samples:%d "
                "measurement_errors:%d solve_fallbacks:%d "
                "weight_norm:%.6g" % (
                    axis, status["training_samples"], status["active"],
                    status["hybrid_solves"],
                    status["prediction_gap_samples"],
                    status["maximum_prediction_gap_samples"],
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
            x_status = self._get_hybrid_axis_status("x")
            y_status = self._get_hybrid_axis_status("y")
            if x_status["active"] and y_status["active"]:
                gcmd.respond_info(
                    "Hybrid FBF ready: x_training:%d y_training:%d"
                    % (x_status["training_samples"],
                       y_status["training_samples"]))
                return
            eventtime = reactor.monotonic()
            if eventtime >= deadline:
                raise gcmd.error(
                    "Hybrid FBF warm-up timed out: x_training:%d "
                    "y_training:%d required:%d"
                    % (x_status["training_samples"],
                       y_status["training_samples"],
                       self.hybrid_warmup_samples))
            reactor.pause(min(deadline, eventtime + .050))

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
            ("hybrid_observation_errors", self.hybrid_observation_errors),
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
