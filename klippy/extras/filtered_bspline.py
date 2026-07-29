# Limited-preview filtered B-spline feedforward controller
#
# Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import collections
import chelper


# Fixed dissertation parameters.  The C implementation uses the same values.
SAMPLE_TIME = 0.001
B_SPLINE_DEGREE = 5
KNOT_SPACING_SAMPLES = 10
BATCH_SAMPLES = 70
PREVIEW_SAMPLES = 140
IMPULSE_RESPONSE_SAMPLES = 20
PREVIEW_TIME = SAMPLE_TIME * PREVIEW_SAMPLES


class FilteredBSpline:
    """Connect the nominal filtered B-spline controller to Klipper motion."""

    def __init__(self, config):
        self.printer = config.get_printer()
        self.initial_enabled = config.getboolean("enabled", False)
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

        self.toolhead = None
        self.enabled = False
        self.stepper_kinematics = []
        self.printer.register_event_handler("klippy:connect", self._connect)
        self.printer.register_event_handler(
            "toolhead:set_position", self._handle_set_position)

        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "SET_FILTERED_BSPLINE", self.cmd_SET_FILTERED_BSPLINE,
            desc=self.cmd_SET_FILTERED_BSPLINE_help)

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

        if not self.stepper_kinematics:
            raise self.printer.config_error(
                "filtered_bspline did not find an X or Y stepper")
        self.toolhead.register_motion_prepare_callback(self._prepare_motion)
        if self.initial_enabled:
            self._set_enabled(True)

    def _prepare_motion(self, start_time, end_time, is_final):
        """Solve every 70-sample batch with a complete 140-sample preview."""
        result = self.ffi_lib.filtered_bspline_prepare(
            self.controller, self.toolhead.get_trapq(), start_time, end_time,
            is_final)
        if result == -2:
            raise self.printer.command_error(
                "Filtered B-spline motion exceeds an axis travel limit")
        if result:
            raise self.printer.command_error(
                "Unable to prepare filtered B-spline motion")

    def _handle_set_position(self):
        """Discard coefficients when homing or G92 changes the coordinates."""
        self.ffi_lib.filtered_bspline_controller_reset(self.controller)

    def _set_enabled(self, enabled):
        """Change controller state only after all old steps are flushed."""
        if enabled == self.enabled:
            return
        old_delay = PREVIEW_TIME if self.enabled else 0.
        new_delay = PREVIEW_TIME if enabled else 0.
        self.toolhead.flush_step_generation()
        self.toolhead.note_step_generation_scan_time(
            new_delay, old_delay=old_delay)
        self.ffi_lib.filtered_bspline_set_enabled(self.controller, enabled)
        self.enabled = enabled

    cmd_SET_FILTERED_BSPLINE_help = (
        "Enable or disable limited-preview filtered B-spline feedforward")

    def cmd_SET_FILTERED_BSPLINE(self, gcmd):
        """Handle SET_FILTERED_BSPLINE ENABLE=0|1."""
        enabled = bool(gcmd.get_int(
            "ENABLE", int(self.enabled), minval=0, maxval=1))
        self._set_enabled(enabled)
        gcmd.respond_info(
            "filtered_bspline enabled:%d sample_time:%.4f preview_time:%.3f"
            % (self.enabled, SAMPLE_TIME, PREVIEW_TIME))

    def get_status(self, eventtime):
        """Expose the fixed experimental design through Klipper status."""
        return collections.OrderedDict([
            ("enabled", self.enabled),
            ("sample_time", SAMPLE_TIME),
            ("degree", B_SPLINE_DEGREE),
            ("knot_spacing_samples", KNOT_SPACING_SAMPLES),
            ("batch_samples", BATCH_SAMPLES),
            ("preview_samples", PREVIEW_SAMPLES),
            ("impulse_response_samples", IMPULSE_RESPONSE_SAMPLES),
        ])


def load_config(config):
    return FilteredBSpline(config)
