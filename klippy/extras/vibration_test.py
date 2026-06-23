# Sinusoidal vibration test support
#
# Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import copy, math


AXIS_INDEX = {'X': 0, 'Y': 1, 'Z': 2}

DEFAULT_AMPLITUDE = 0.02
DEFAULT_DURATION = 5.0
DEFAULT_FREQUENCY = 40.0
DEFAULT_SEGMENTS_PER_CYCLE = 32
DEFAULT_MAX_SEGMENT_RATE = 6000.0
DEFAULT_CHUNK_TIME = 0.250
DEFAULT_RAMP_CYCLES = 2.0
DEFAULT_MAX_SEGMENTS = 100000


class ValidationMove:
    def __init__(self, printer, max_accel, start_pos, end_pos, speed):
        self.printer = printer
        self.start_pos = tuple(start_pos)
        self.end_pos = tuple(end_pos)
        self.axes_d = [end_pos[i] - start_pos[i] for i in (0, 1, 2, 3)]
        self.move_d = math.sqrt(sum([d * d for d in self.axes_d[:3]]))
        self.axes_r = [0., 0., 0., 0.]
        if self.move_d:
            inv_move_d = 1. / self.move_d
            self.axes_r = [d * inv_move_d for d in self.axes_d]
        self.max_cruise_v2 = speed * speed
        self.accel = max_accel

    def limit_speed(self, speed, accel):
        self.max_cruise_v2 = min(self.max_cruise_v2, speed * speed)
        self.accel = min(self.accel, accel)

    def move_error(self, msg="Move out of range"):
        ep = self.end_pos
        return self.printer.command_error(
            "%s: %.3f %.3f %.3f [%.3f]"
            % (msg, ep[0], ep[1], ep[2], ep[3]))


class VibrationTest:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.reactor = self.printer.get_reactor()
        self.gcode.register_command(
            "RUN_VIBRATION_TEST", self.cmd_RUN_VIBRATION_TEST,
            desc=self.cmd_RUN_VIBRATION_TEST_help)

    def _get_axis(self, gcmd):
        axis = gcmd.get('AXIS', default='X').upper()
        if axis not in AXIS_INDEX:
            raise gcmd.error(
                '{"code":"key274", "msg": "Invalid axis: %s", '
                '"values": ["%s"]}' % (axis, axis))
        return axis, AXIS_INDEX[axis]

    def _get_ramp_time(self, gcmd, duration, frequency):
        ramp_time = gcmd.get_float('RAMP_TIME', None, minval=0.,
                                   maxval=duration * 0.5)
        if ramp_time is not None:
            return ramp_time
        ramp_cycles = gcmd.get_float('RAMP_CYCLES', DEFAULT_RAMP_CYCLES,
                                     minval=0.)
        return min(duration * 0.25, ramp_cycles / frequency)

    def _envelope(self, t, duration, ramp_time):
        if ramp_time <= 0.:
            return 1.
        if t < ramp_time:
            u = t / ramp_time
        elif t > duration - ramp_time:
            u = (duration - t) / ramp_time
        else:
            return 1.
        u = max(0., min(1., u))
        # Smootherstep keeps velocity and acceleration continuous at endpoints.
        return u * u * u * (u * (u * 6. - 15.) + 10.)

    def _wave_offset(self, t, duration, frequency, amplitude, ramp_time):
        omega = 2. * math.pi * frequency
        return (amplitude * self._envelope(t, duration, ramp_time)
                * math.sin(omega * t))

    def _iter_segments(self, duration, frequency, amplitude, ramp_time,
                       segment_time):
        segment_count = int(math.ceil(duration / segment_time))
        for i in range(segment_count):
            t0 = min(duration, i * segment_time)
            t1 = min(duration, (i + 1) * segment_time)
            dt = t1 - t0
            if dt <= 0.:
                continue
            tm = t0 + 0.5 * dt
            x0 = self._wave_offset(t0, duration, frequency, amplitude,
                                   ramp_time)
            xm = self._wave_offset(tm, duration, frequency, amplitude,
                                   ramp_time)
            x1 = self._wave_offset(t1, duration, frequency, amplitude,
                                   ramp_time)
            accel = 4. * (x1 - 2. * xm + x0) / (dt * dt)
            start_v = (x1 - x0) / dt - 0.5 * accel * dt
            yield t0, dt, x0, start_v, accel, x1

    def _scan_segments(self, duration, frequency, amplitude, ramp_time,
                       segment_time):
        max_v = max_a = 0.
        min_x, max_x = -abs(amplitude), abs(amplitude)
        count = 0
        for t0, dt, x0, start_v, accel, x1 in self._iter_segments(
                duration, frequency, amplitude, ramp_time, segment_time):
            end_v = start_v + accel * dt
            max_v = max(max_v, abs(start_v), abs(end_v))
            max_a = max(max_a, abs(accel))
            min_x = min(min_x, x0, x1)
            max_x = max(max_x, x0, x1)
            count += 1
        return count, min_x, max_x, max_v, max_a

    def _check_motion_limits(self, gcmd, toolhead, axis_index, start_pos,
                             min_x, max_x, max_v, max_a):
        systime = self.reactor.monotonic()
        toolhead_info = toolhead.get_status(systime)
        axis = "XYZ"[axis_index].lower()
        if axis not in toolhead_info['homed_axes']:
            raise gcmd.error(
                "Must home %s axis before running vibration test"
                % (axis.upper(),))

        axis_min = getattr(toolhead_info['axis_minimum'], axis)
        axis_max = getattr(toolhead_info['axis_maximum'], axis)
        test_min = start_pos[axis_index] + min_x
        test_max = start_pos[axis_index] + max_x
        if test_min < axis_min or test_max > axis_max:
            raise gcmd.error(
                "Vibration test would move %s outside range: "
                "%.6f..%.6f not within %.6f..%.6f"
                % (axis.upper(), test_min, test_max, axis_min, axis_max))

        kin = toolhead.get_kinematics()
        check_speed = max(max_v, 0.000001)
        saved_kin_state = self._save_kinematics_state(kin)
        try:
            for offset in (min_x, max_x):
                if not offset:
                    continue
                end_pos = list(start_pos)
                end_pos[axis_index] += offset
                move = ValidationMove(self.printer, toolhead.max_accel,
                                      start_pos, end_pos, check_speed)
                kin.check_move(move)
        finally:
            self._restore_kinematics_state(kin, saved_kin_state)

        max_velocity = gcmd.get_float(
            'MAX_VELOCITY', toolhead_info['max_velocity'], above=0.)
        max_accel = gcmd.get_float(
            'MAX_ACCEL', toolhead_info['max_accel'], above=0.)
        if axis_index == 2:
            max_velocity = min(max_velocity,
                               getattr(kin, 'max_z_velocity', max_velocity))
            max_accel = min(max_accel, getattr(kin, 'max_z_accel', max_accel))
        if max_v > max_velocity * 1.000001:
            raise gcmd.error(
                "Requested vibration requires %.3f mm/s, limit is %.3f mm/s"
                % (max_v, max_velocity))
        if max_a > max_accel * 1.000001:
            raise gcmd.error(
                "Requested vibration requires %.3f mm/s^2, limit is %.3f "
                "mm/s^2" % (max_a, max_accel))

    def _save_kinematics_state(self, kin):
        saved = {}
        for name in ('limit_xy2', 'limit_z', 'limits',
                     'need_home', 'homed_axis'):
            if hasattr(kin, name):
                saved[name] = copy.deepcopy(getattr(kin, name))
        return saved

    def _restore_kinematics_state(self, kin, saved):
        for name, value in saved.items():
            setattr(kin, name, value)

    def _enter_direct_motion(self, toolhead):
        toolhead.flush_step_generation()
        if toolhead.special_queuing_state:
            if toolhead.special_queuing_state == "Drip":
                raise self.printer.command_error(
                    "Can not run vibration test during drip move")
            toolhead.special_queuing_state = ""
            toolhead.need_check_stall = -1.
            toolhead.reactor.update_timer(toolhead.flush_timer,
                                          toolhead.reactor.NOW)
            toolhead._calc_print_time()
        return toolhead.print_time

    def _append_segment(self, toolhead, print_time, start_pos, axis_index,
                        offset, dt, start_v, accel):
        pos = list(start_pos)
        axes_r = [0., 0., 0.]
        pos[axis_index] += offset
        axes_r[axis_index] = 1.
        toolhead.trapq_append(
            toolhead.trapq, print_time,
            dt, 0., 0.,
            pos[0], pos[1], pos[2],
            axes_r[0], axes_r[1], axes_r[2],
            start_v, 0., accel)

    def _set_commanded_axis_pos(self, toolhead, start_pos, axis_index, offset):
        pos = list(start_pos)
        if abs(offset) < 0.000000001:
            offset = 0.
        pos[axis_index] += offset
        toolhead.commanded_pos[:] = pos

    def _run_direct_sinusoid(self, toolhead, axis_index, duration, frequency,
                             amplitude, ramp_time, segment_time, chunk_time):
        start_pos = toolhead.get_position()
        print_time = self._enter_direct_motion(toolhead)
        chunk_start = print_time

        for t0, dt, x0, start_v, accel, x1 in self._iter_segments(
                duration, frequency, amplitude, ramp_time, segment_time):
            self._append_segment(toolhead, print_time, start_pos, axis_index,
                                 x0, dt, start_v, accel)
            print_time += dt
            if print_time - chunk_start >= chunk_time:
                toolhead._update_move_time(print_time)
                toolhead.last_kin_move_time = print_time
                self._set_commanded_axis_pos(
                    toolhead, start_pos, axis_index, x1)
                toolhead._check_stall()
                chunk_start = print_time

        if toolhead.print_time < print_time:
            toolhead._update_move_time(print_time)
        toolhead.last_kin_move_time = print_time
        final_offset = self._wave_offset(duration, duration, frequency,
                                         amplitude, ramp_time)
        self._set_commanded_axis_pos(toolhead, start_pos, axis_index,
                                     final_offset)
        toolhead._check_stall()

    cmd_RUN_VIBRATION_TEST_help = (
        "Run streamed sinusoidal motion. Params: AXIS=<X|Y|Z> "
        "DURATION=<0.1..10s> FREQUENCY=<Hz> AMPLITUDE=<mm> "
        "SEGMENTS_PER_CYCLE=<8..200> MAX_SEGMENT_RATE=<segments/s> "
        "CHUNK_TIME=<0.05..1s> MAX_SEGMENTS=<count> RAMP_TIME=<s> "
        "RAMP_CYCLES=<cycles> MAX_VELOCITY=<mm/s> "
        "MAX_ACCEL=<mm/s^2> INPUT_SHAPING=<0|1> WAIT=<0|1>")

    def cmd_RUN_VIBRATION_TEST(self, gcmd):
        axis, axis_index = self._get_axis(gcmd)
        duration = gcmd.get_float('DURATION', DEFAULT_DURATION,
                                  minval=0.100, maxval=10.)
        frequency = gcmd.get_float('FREQUENCY', DEFAULT_FREQUENCY,
                                   above=0., maxval=1000.)
        amplitude = gcmd.get_float('AMPLITUDE', DEFAULT_AMPLITUDE, above=0.)
        segments_per_cycle = gcmd.get_int(
            'SEGMENTS_PER_CYCLE', DEFAULT_SEGMENTS_PER_CYCLE,
            minval=8, maxval=200)
        max_segment_rate = gcmd.get_float(
            'MAX_SEGMENT_RATE', DEFAULT_MAX_SEGMENT_RATE,
            above=0., maxval=20000.)
        chunk_time = gcmd.get_float('CHUNK_TIME', DEFAULT_CHUNK_TIME,
                                    minval=0.050, maxval=1.000)
        max_segments = gcmd.get_int('MAX_SEGMENTS', DEFAULT_MAX_SEGMENTS,
                                    minval=1)
        wait = gcmd.get_int('WAIT', 1, minval=0, maxval=1)

        segment_rate = frequency * segments_per_cycle
        segment_time = 1. / segment_rate
        if segment_rate > max_segment_rate:
            raise gcmd.error(
                "Requested segment rate %.0f/s exceeds MAX_SEGMENT_RATE %.0f/s"
                % (segment_rate, max_segment_rate))
        ramp_time = self._get_ramp_time(gcmd, duration, frequency)
        if ramp_time <= 0. and abs(duration * frequency
                                  - round(duration * frequency)) > 0.000001:
            raise gcmd.error(
                "RAMP_TIME=0 requires DURATION*FREQUENCY to be an integer")

        count, min_x, max_x, max_v, max_a = self._scan_segments(
            duration, frequency, amplitude, ramp_time, segment_time)
        if count > max_segments:
            raise gcmd.error(
                "Vibration test would generate %d segments, limit is %d"
                % (count, max_segments))

        toolhead = self.printer.lookup_object('toolhead')
        start_pos = toolhead.get_position()
        self._check_motion_limits(gcmd, toolhead, axis_index, start_pos,
                                  min_x, max_x, max_v, max_a)

        input_shaper = self.printer.lookup_object('input_shaper', None)
        if input_shaper is not None and not gcmd.get_int('INPUT_SHAPING', 0):
            input_shaper.disable_shaping()
            gcmd.respond_info("Disabled [input_shaper] for vibration test")
        else:
            input_shaper = None

        try:
            gcmd.respond_info(
                "Vibration test %s: %.3f Hz, %.6f mm, %.3f s, "
                "%d segments (%.0f/s), peak %.3f mm/s, %.3f mm/s^2"
                % (axis, frequency, amplitude, duration, count, segment_rate,
                   max_v, max_a))
            self._run_direct_sinusoid(toolhead, axis_index, duration,
                                      frequency, amplitude, ramp_time,
                                      segment_time, chunk_time)
            if wait:
                toolhead.wait_moves()
        finally:
            if input_shaper is not None:
                input_shaper.enable_shaping()
                gcmd.respond_info("Re-enabled [input_shaper]")


def load_config(config):
    return VibrationTest(config)
