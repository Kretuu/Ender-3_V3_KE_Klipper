# Continuous Trinkey/BNO055 accelerometer streaming
#
# Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections, logging, re, struct, threading, time, zlib

import serial

import chelper
from . import motion_report


STREAM_MAGIC = 0x32534b54  # little-endian bytes: TKS2
STREAM_FRAME_DATA = 1
STREAM_FRAME_SYNC = 2
STREAM_FRAME_STOP = 3
STREAM_FRAME_STATUS = 4
STREAM_PROTOCOL_VERSION = 2

STREAM_DATA_HEADER = struct.Struct('<IHBBIIIIIII')
STREAM_SAMPLE = struct.Struct('<IQhhhQhhhB')
STREAM_SYNC = struct.Struct('<IQ')
STREAM_STATUS = struct.Struct('<IIBIIIIIII')

FLAG_BASE_VALID = 0x01
FLAG_TOOLHEAD_VALID = 0x02
FLAG_BASE_ERROR = 0x04
FLAG_TOOLHEAD_ERROR = 0x08

ACK_STREAM_STOP_RE = re.compile(r'ACK_STREAM_STOP,(\d+)')
ACK_STREAM_START_RE = re.compile(
    r'ACK_STREAM_START,([^,\r\n]*),(base|toolhead|both),(\d+),(\d+),'
    r'(\d+),(\d+)')
SYNC_RE = re.compile(r'SYNC,(\d+),(\d+)')
ACK_BW_RE = re.compile(r'ACK_BW,(\d+),(\d+)')

SERIAL_BAUD = 115200
CONTROL_TIMEOUT = 5.0
SYNC_TIMEOUT = 2.0
CONTROL_RETRY_INTERVAL = 0.500
SERIAL_RECORD_LIMIT = 65536
DEFAULT_API_UPDATE_INTERVAL = 0.100
DEFAULT_SYNC_INTERVAL = 1.0
DEFAULT_INITIAL_SYNC_SAMPLES = 12
DEFAULT_RAW_SCALE = 10.0  # BNO055 default m/s^2 units: 100 LSB/(m/s^2)
SYNC_RTT_TOLERANCE = 0.001  # Experiment-specific USB RTT margin, seconds.
TRAPQ_HISTORY_SAFE_AGE = 25.0  # trapq.c retains 30 seconds of history.
OBSERVER_TIME_TOLERANCE = 0.25


def _sample_trapq_motion(moves, timed_keys):
    """Return nominal XYZ position and acceleration at sorted times."""
    results = {}
    move_index = 0
    for print_time, key in timed_keys:
        while move_index < len(moves):
            move = moves[move_index]
            if print_time < move.print_time + move.move_t:
                break
            move_index += 1
        if move_index >= len(moves):
            results[key] = ((0., 0., 0.), (0., 0., 0.), False)
            continue
        move = moves[move_index]
        if print_time < move.print_time:
            results[key] = (
                (move.start_x, move.start_y, move.start_z),
                (0., 0., 0.), True)
            continue
        move_time = min(move.move_t, print_time - move.print_time)
        distance = (move.start_v + .5 * move.accel * move_time) * move_time
        position = (
            move.start_x + move.x_r * distance,
            move.start_y + move.y_r * distance,
            move.start_z + move.z_r * distance)
        acceleration = (
            move.accel * move.x_r,
            move.accel * move.y_r,
            move.accel * move.z_r)
        results[key] = (position, acceleration, True)
    return results


def _cobs_decode(data):
    """Decode one COBS frame without its trailing zero delimiter."""
    output = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if code == 0:
            raise ValueError("zero byte inside COBS frame")
        index += 1
        end = index + code - 1
        if end > len(data):
            raise ValueError("truncated COBS frame")
        output.extend(data[index:end])
        index = end
        if code != 0xff and index < len(data):
            output.append(0)
    return bytes(output)


class TrinkeyClockMapper:
    """Fit Trinkey microseconds directly to primary-MCU print time."""
    def __init__(self, max_points=64):
        self.points = collections.deque(maxlen=max_points)
        self.slope = 1.0
        self.offset = 0.0
        self.last_rtt = 0.0
        self.last_device_time_us = 0
        self.last_print_time = 0.0
        self.accepted_points = 0
        self.rtt_minimum = 0.0
        self.rtt_limit = 0.0
        self.ready = False

    def reset(self):
        self.points.clear()
        self.slope = 1.0
        self.offset = 0.0
        self.last_rtt = 0.0
        self.last_device_time_us = 0
        self.last_print_time = 0.0
        self.accepted_points = 0
        self.rtt_minimum = 0.0
        self.rtt_limit = 0.0
        self.ready = False

    def add(self, device_time_us, print_time, rtt):
        device_time = float(device_time_us) * 1.e-6
        self.points.append((device_time, float(print_time), float(rtt)))
        self.last_rtt = float(rtt)
        self.last_device_time_us = int(device_time_us)
        self.last_print_time = float(print_time)
        self._fit()

    def _fit(self):
        if not self.points:
            return
        self.rtt_minimum = min(p[2] for p in self.points)
        self.rtt_limit = self.rtt_minimum + SYNC_RTT_TOLERANCE
        points = [p for p in self.points if p[2] <= self.rtt_limit]
        self.accepted_points = len(points)

        # Until the clock observations span several seconds, estimating only
        # the offset is substantially less sensitive to USB RTT jitter.
        offset = sum(p[1] - p[0] for p in points) / len(points)
        slope = 1.0
        if len(points) >= 8 and points[-1][0] - points[0][0] >= 5.0:
            mean_x = sum(p[0] for p in points) / len(points)
            mean_y = sum(p[1] for p in points) / len(points)
            variance = sum((p[0] - mean_x) ** 2 for p in points)
            if variance > 0.:
                candidate = sum(
                    (p[0] - mean_x) * (p[1] - mean_y)
                    for p in points) / variance
                if 0.995 <= candidate <= 1.005:
                    slope = candidate
                    offset = mean_y - slope * mean_x
        self.slope = slope
        self.offset = offset
        self.ready = True

    def get_print_time(self, device_time_us):
        return self.offset + self.slope * (float(device_time_us) * 1.e-6)

    def get_status(self):
        return {
            'ready': self.ready,
            'points': len(self.points),
            'accepted_points': self.accepted_points,
            'rtt_minimum': self.rtt_minimum,
            'rtt_limit': self.rtt_limit,
            'rtt_tolerance': SYNC_RTT_TOLERANCE,
            'slope': self.slope,
            'offset': self.offset,
            'last_rtt': self.last_rtt,
            'last_device_time_us': self.last_device_time_us,
            'last_print_time': self.last_print_time,
        }


class TrinkeySensorDump:
    def __init__(self, parent, sensor, update_interval):
        self.parent = parent
        self.sensor = sensor
        self.api_dump = motion_report.APIDumpHelper(
            parent.printer, self._api_update, self._api_startstop,
            update_interval)
        webhooks = parent.printer.lookup_object('webhooks')
        webhooks.register_mux_endpoint(
            'trinkey_accel/dump_trinkey_accel', 'sensor', sensor,
            self._handle_dump)

    def _api_startstop(self, is_start):
        if is_start:
            self.parent.add_stream_client()
        else:
            self.parent.remove_stream_client()

    def _api_update(self, eventtime):
        return self.parent.api_update(self.sensor, eventtime)

    def _handle_dump(self, web_request):
        self.api_dump.add_client(web_request)
        header = (
            'time', 'device_time_us', 'sample_sequence',
            'x_raw', 'y_raw', 'z_raw',
            'x_acceleration', 'y_acceleration', 'z_acceleration', 'flags')
        web_request.send({'header': header})

    def start_internal_client(self):
        return self.api_dump.add_internal_client()


class TrinkeyExperimentDump:
    """Combined sensor and sampled nominal-motion stream for experiments."""
    def __init__(self, parent, update_interval):
        self.parent = parent
        self.api_dump = motion_report.APIDumpHelper(
            parent.printer, self._api_update, self._api_startstop,
            update_interval)
        webhooks = parent.printer.lookup_object('webhooks')
        webhooks.register_endpoint(
            'trinkey_accel/dump_experiment', self._handle_dump)

    def _api_startstop(self, is_start):
        if is_start:
            self.parent.add_stream_client()
        else:
            self.parent.remove_stream_client()

    def _api_update(self, eventtime):
        return self.parent.api_update_experiment(eventtime)

    def _handle_dump(self, web_request):
        self.api_dump.add_client(web_request)
        common_header = (
            'time', 'device_time_us', 'sample_sequence',
            'x_raw', 'y_raw', 'z_raw',
            'x_acceleration', 'y_acceleration', 'z_acceleration', 'flags')
        web_request.send({
            'headers': {
                'base': common_header + (
                    'desired_y_position', 'desired_y_acceleration',
                    'motor_y_position', 'motor_y_velocity',
                    'observed_y_position', 'observer_y_acceleration',
                    'observer_valid', 'reference_valid'),
                'toolhead': common_header + (
                    'desired_x_position', 'desired_x_acceleration',
                    'motor_x_position', 'motor_x_velocity',
                    'observed_x_position', 'observer_x_acceleration',
                    'observer_valid', 'reference_valid'),
            },
            'reference': 'nominal_trapq_and_final_step_command_at_sample_time',
        })

    def start_internal_client(self):
        """Give future hybrid control the same batches sent to Motan."""
        return self.api_dump.add_internal_client()


class TrinkeyAccel:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.mcu = self.printer.lookup_object('mcu')
        self.port = config.get('serial')
        self.rate = config.getint('rate', 250, minval=100, maxval=1000)
        if 1000000 % self.rate:
            raise config.error(
                "[trinkey_accel] rate must divide 1000000 exactly")
        self.batch_size = config.getint(
            'batch_size', 4, minval=1, maxval=16)
        self.bandwidth = config.getint('bandwidth', 125)
        if self.bandwidth not in (8, 16, 31, 63, 125, 250, 500, 1000):
            raise config.error(
                "[trinkey_accel] bandwidth must be one of "
                "8, 16, 31, 63, 125, 250, 500, or 1000 Hz")
        sensors = config.getlist(
            'sensors', ('base', 'toolhead'))
        self.sensors = tuple(s.strip().lower() for s in sensors)
        if (not self.sensors
                or any(s not in ('base', 'toolhead') for s in self.sensors)
                or len(set(self.sensors)) != len(self.sensors)):
            raise config.error(
                "[trinkey_accel] sensors must contain base and/or toolhead")
        self.mode = ('both' if len(self.sensors) == 2
                     else self.sensors[0])
        self.raw_scale = config.getfloat(
            'raw_lsb_mm_s2', DEFAULT_RAW_SCALE, above=0.)
        self.toolhead = None
        self.kinematics = None
        self.motion_steppers = []
        self.observer_axes = {}
        self.observer_resets = 0
        self.motion_lookup_errors = 0
        self._configure_observers(config)
        self.sync_interval = config.getfloat(
            'sync_interval', DEFAULT_SYNC_INTERVAL, above=0.1)
        self.initial_sync_samples = config.getint(
            'initial_sync_samples', DEFAULT_INITIAL_SYNC_SAMPLES,
            minval=2, maxval=50)
        api_update_interval = config.getfloat(
            'api_update_interval', DEFAULT_API_UPDATE_INTERVAL,
            above=0.01, maxval=1.0)
        self.max_host_samples = config.getint(
            'max_host_samples', 20000, minval=1000)

        self.lifecycle_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.control_condition = threading.Condition()
        self.data_lock = threading.Lock()
        self.serial_conn = None
        self.reader_thread = None
        self.reader_running = False
        self.binary_mode = False
        self.reader_error = None
        self.stream_clients = 0
        self.streaming = False
        self.stream_start_pending = False
        self.ack_stream_stop = None
        self.ack_stream_start = None
        self.ack_bandwidth = None
        self.stream_stopped = False
        self.idle_sync_responses = {}
        self.stream_sync_pending = {}
        self.stream_sync_responses = []
        self.next_sync_sequence = 1
        self.sync_timer = None
        self.sync_timeouts = 0
        self.clock_mapper = TrinkeyClockMapper()
        self.sample_queues = {
            sensor: collections.deque(maxlen=self.max_host_samples)
            for sensor in ('base', 'toolhead')
        }

        self.host_queue_drops = 0
        self.frame_errors = 0
        self.packet_sequence_errors = 0
        self.last_packet_sequence = None
        self.read_errors = 0
        self.firmware_status = {}

        self.sensor_dumps = {
            sensor: TrinkeySensorDump(self, sensor, api_update_interval)
            for sensor in ('base', 'toolhead')
        }
        self.experiment_dump = TrinkeyExperimentDump(
            self, api_update_interval)
        gcode = self.printer.lookup_object('gcode')
        gcode.register_command(
            'TRINKEY_ACCEL_STATUS', self.cmd_TRINKEY_ACCEL_STATUS,
            desc=self.cmd_TRINKEY_ACCEL_STATUS_help)
        self.printer.register_event_handler(
            'klippy:connect', self._handle_connect)
        self.printer.register_event_handler(
            'klippy:shutdown', self._handle_shutdown)
        self.printer.register_event_handler(
            'klippy:disconnect', self._handle_shutdown)

    def _configure_observers(self, config):
        """Load the offline-designed runtime matrices for both axes."""
        if not config.getboolean('observer_enabled', False):
            return
        observer_rate = config.getint(
            'observer_sample_rate', minval=100, maxval=1000)
        if observer_rate != self.rate:
            raise config.error(
                "[trinkey_accel] observer_sample_rate must equal rate")
        bias_samples = config.getint(
            'observer_bias_samples', 100, minval=0, maxval=10000)
        signs = {
            'x': config.getfloat('observer_x_accel_sign', -1.,
                                 minval=-1., maxval=1.),
            'y': config.getfloat('observer_y_accel_sign', 1.,
                                 minval=-1., maxval=1.),
        }
        if signs['x'] not in (-1., 1.) or signs['y'] not in (-1., 1.):
            raise config.error(
                "[trinkey_accel] observer acceleration signs must be -1 or 1")

        self.ffi_main, self.ffi_lib = chelper.get_ffi()
        for axis in ('x', 'y'):
            prefix = 'observer_%s_' % (axis,)
            state_count = config.getint(
                prefix + 'state_count', minval=1, maxval=8)
            fo = config.getfloatlist(
                prefix + 'fo', count=state_count * state_count)
            gu = config.getfloatlist(prefix + 'gu', count=state_count)
            gv = config.getfloatlist(prefix + 'gv', count=state_count)
            ga = config.getfloatlist(prefix + 'ga', count=state_count)
            ho = config.getfloatlist(prefix + 'ho', count=state_count)
            x0 = config.getfloatlist(
                prefix + 'x0_per_mm', count=state_count)
            observer = self.ffi_lib.state_space_observer_alloc(
                state_count, fo, gu, gv, ga, ho, x0)
            if observer == self.ffi_main.NULL:
                raise config.error(
                    "Invalid %s-axis state-space observer matrices"
                    % (axis.upper(),))
            self.observer_axes[axis] = {
                'c': self.ffi_main.gc(
                    observer, self.ffi_lib.state_space_observer_free),
                'sign': signs[axis],
                'sample_time': 1. / observer_rate,
                'bias_samples': bias_samples,
            }
        self._reset_observers()

    def _reset_observers(self):
        """Discard state and repeat stationary bias calibration."""
        for observer in self.observer_axes.values():
            self.ffi_lib.state_space_observer_reset(observer['c'])
            observer.update({
                'bias_count': 0,
                'bias_sum': 0.,
                'bias': 0.,
                'bias_ready': observer['bias_samples'] == 0,
                'last_time': None,
                'last_position': 0.,
            })

    def _handle_connect(self):
        """Keep the generic kinematics used to invert final step positions."""
        self.toolhead = self.printer.lookup_object('toolhead')
        self.kinematics = self.toolhead.get_kinematics()
        self.motion_steppers = self.kinematics.get_steppers()

    def _get_commanded_position(self, print_time):
        """Reconstruct final commanded XYZ from read-only step history."""
        stepper_positions = {}
        for stepper in self.motion_steppers:
            mcu_position = stepper.get_past_mcu_position(print_time)
            stepper_positions[stepper.get_name()] = (
                stepper.mcu_to_commanded_position(mcu_position))
        return self.kinematics.calc_position(stepper_positions)

    def _reset_axis_state(self, observer):
        """Reset dynamic state after a timing gap, retaining sensor bias."""
        if observer['last_time'] is None:
            return
        self.ffi_lib.state_space_observer_reset(observer['c'])
        observer['last_time'] = None
        observer['last_position'] = 0.
        self.observer_resets += 1

    def _observe_axis(self, axis, print_time, position, measured_acceleration):
        """Run one fixed-rate observer sample and return logged scalars."""
        observer = self.observer_axes.get(axis)
        if observer is None:
            return 0., 0., 0., 0

        acceleration = observer['sign'] * measured_acceleration
        if not observer['bias_ready']:
            observer['bias_sum'] += acceleration
            observer['bias_count'] += 1
            if observer['bias_count'] < observer['bias_samples']:
                return 0., 0., 0., 0
            observer['bias'] = (
                observer['bias_sum'] / observer['bias_count'])
            observer['bias_ready'] = True

        corrected_acceleration = acceleration - observer['bias']
        last_time = observer['last_time']
        velocity = 0.
        valid = 0
        if last_time is not None:
            sample_time = print_time - last_time
            expected_time = observer['sample_time']
            timing_error = abs(sample_time - expected_time) / expected_time
            if sample_time <= 0. or timing_error > OBSERVER_TIME_TOLERANCE:
                self._reset_axis_state(observer)
                valid = 0
            else:
                velocity = (
                    position - observer['last_position']) / sample_time
                valid = 1

        estimated_position = self.ffi_lib.state_space_observer_sample(
            observer['c'], position, velocity, corrected_acceleration)
        observer['last_time'] = print_time
        observer['last_position'] = position
        return velocity, estimated_position, corrected_acceleration, valid

    def _command_error(self, message):
        return self.printer.command_error(message)

    def _send_command(self, command):
        if self.serial_conn is None:
            raise self._command_error("Trinkey serial port is not open")
        data = (command + '\n').encode('ascii')
        with self.write_lock:
            self.serial_conn.write(data)
            self.serial_conn.flush()

    def _wait_for(
            self, predicate, timeout, message, raise_on_timeout=True):
        deadline = self.reactor.monotonic() + timeout
        with self.control_condition:
            while not predicate():
                if self.reader_error is not None:
                    raise self._command_error(
                        "Trinkey reader failed: %s" % (self.reader_error,))
                remaining = deadline - self.reactor.monotonic()
                if remaining <= 0.:
                    if not raise_on_timeout:
                        return False
                    raise self._command_error(message)
                self.control_condition.wait(remaining)
            return True

    def _open_serial(self):
        try:
            self.serial_conn = serial.Serial(
                self.port, SERIAL_BAUD, timeout=0.050, write_timeout=0.500)
            self.serial_conn.reset_input_buffer()
        except Exception as e:
            raise self._command_error(
                "Unable to open Trinkey serial port %s: %s"
                % (self.port, e))
        self.reader_error = None
        self.binary_mode = False

    def _start_reader(self):
        self.reader_running = True
        self.reader_thread = threading.Thread(target=self._reader_loop)
        self.reader_thread.daemon = True
        self.reader_thread.start()

    def _close_serial(self):
        self.reader_running = False
        if self.reader_thread is not None:
            self.reader_thread.join(0.500)
            self.reader_thread = None
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                logging.exception("Unable to close Trinkey serial port")
            self.serial_conn = None
        self.binary_mode = False
        self.streaming = False
        self.stream_start_pending = False

    def _recover_idle(self):
        # The previous logger may have exited while the firmware was emitting
        # zero-delimited binary frames. Recover before starting the reader so
        # that stale binary data can never be parsed as newline-delimited
        # control responses. STREAM_STOP is idempotent and its ASCII response
        # follows the final binary STOP frame when a stream was active.
        self.ack_stream_stop = None
        response = bytearray()
        deadline = self.reactor.monotonic() + CONTROL_TIMEOUT
        next_request = 0.
        while self.reactor.monotonic() < deadline:
            now = self.reactor.monotonic()
            if now >= next_request:
                self._send_command('STREAM_STOP')
                next_request = now + CONTROL_RETRY_INTERVAL
            available = self.serial_conn.in_waiting
            chunk = self.serial_conn.read(min(available, 4096) or 1)
            if not chunk:
                continue
            response.extend(chunk)
            if len(response) > SERIAL_RECORD_LIMIT:
                del response[:-SERIAL_RECORD_LIMIT]
            match = ACK_STREAM_STOP_RE.search(
                response.decode('utf-8', 'replace'))
            if match is not None:
                self.ack_stream_stop = match.group(0)
                self.serial_conn.reset_input_buffer()
                return
        raise self._command_error("Timed out putting Trinkey into idle mode")

    def _idle_sync_once(self):
        sequence = self.next_sync_sequence
        self.next_sync_sequence += 1
        host_before = self.reactor.monotonic()
        self._send_command('SYNC,%d' % (sequence,))
        self._wait_for(
            lambda: sequence in self.idle_sync_responses,
            SYNC_TIMEOUT,
            "Timed out waiting for Trinkey SYNC %d" % (sequence,))
        with self.control_condition:
            device_time_us, host_after = self.idle_sync_responses.pop(sequence)
        host_mid = .5 * (host_before + host_after)
        print_time = self.mcu.estimated_print_time(host_mid)
        self.clock_mapper.add(
            device_time_us, print_time, host_after - host_before)

    def _start_stream(self):
        self.clock_mapper.reset()
        self._reset_observers()
        with self.data_lock:
            for queue in self.sample_queues.values():
                queue.clear()
        with self.control_condition:
            self.stream_sync_pending.clear()
            self.stream_sync_responses = []
            self.idle_sync_responses.clear()
        self.host_queue_drops = 0
        self.frame_errors = 0
        self.packet_sequence_errors = 0
        self.last_packet_sequence = None
        self.read_errors = 0
        self.observer_resets = 0
        self.motion_lookup_errors = 0
        self.sync_timeouts = 0
        self.firmware_status = {}
        self._open_serial()
        try:
            self._recover_idle()
            self._start_reader()
            with self.control_condition:
                self.ack_bandwidth = None
            self._send_command('SET_BW,%d' % (self.bandwidth,))
            self._wait_for(
                lambda: self.ack_bandwidth is not None,
                CONTROL_TIMEOUT,
                "Timed out configuring Trinkey accelerometer bandwidth")
            if self.ack_bandwidth != self.bandwidth:
                raise self._command_error(
                    "Trinkey acknowledged unexpected bandwidth %s"
                    % (self.ack_bandwidth,))
            for unused in range(self.initial_sync_samples):
                self._idle_sync_once()
                time.sleep(0.010)

            with self.control_condition:
                self.ack_stream_start = None
                self.stream_stopped = False
            run_id = 'motan_%d' % (int(time.time()),)
            self.stream_start_pending = True
            self._send_command(
                'STREAM_START,%s,%s,%d,%d'
                % (run_id, self.mode, self.rate, self.batch_size))
            self._wait_for(
                lambda: self.ack_stream_start is not None,
                CONTROL_TIMEOUT,
                "Timed out starting Trinkey stream")
            ack = self.ack_stream_start
            if (ack[1] != self.mode or int(ack[2]) != self.rate
                    or int(ack[3]) != self.batch_size
                    or int(ack[4]) != STREAM_PROTOCOL_VERSION):
                raise self._command_error(
                    "Unexpected Trinkey stream acknowledgement: %s"
                    % (ack,))
            self.streaming = True
            self.stream_start_pending = False
            self.sync_timer = self.reactor.register_timer(
                self._sync_timer_event,
                self.reactor.monotonic() + self.sync_interval)
            logging.info(
                "Trinkey accelerometer stream started: %s at %d Hz, "
                "batch=%d", self.mode, self.rate, self.batch_size)
        except Exception:
            self._stop_stream()
            raise

    def _request_stream_stop(self):
        with self.control_condition:
            self.stream_stopped = False
            self.ack_stream_stop = None
        deadline = self.reactor.monotonic() + CONTROL_TIMEOUT
        while self.reactor.monotonic() < deadline:
            self._send_command('STREAM_STOP')
            remaining = min(
                CONTROL_RETRY_INTERVAL,
                deadline - self.reactor.monotonic())
            if self._wait_for(
                    lambda: self.ack_stream_stop is not None, remaining,
                    "Timed out stopping Trinkey stream",
                    raise_on_timeout=False):
                return
        raise self._command_error("Timed out stopping Trinkey stream")

    def _stop_stream(self):
        if self.sync_timer is not None:
            self.reactor.unregister_timer(self.sync_timer)
            self.sync_timer = None
        if (self.serial_conn is not None
                and (self.streaming or self.stream_start_pending)):
            try:
                self._request_stream_stop()
            except Exception:
                logging.exception("Unable to stop Trinkey stream cleanly")
        self._close_serial()
        logging.info("Trinkey accelerometer stream stopped")

    def add_stream_client(self):
        with self.lifecycle_lock:
            self.stream_clients += 1
            if self.stream_clients > 1:
                return
            try:
                self._start_stream()
            except Exception:
                self.stream_clients = 0
                raise

    def remove_stream_client(self):
        with self.lifecycle_lock:
            if self.stream_clients <= 0:
                return
            self.stream_clients -= 1
            if self.stream_clients == 0:
                self._stop_stream()

    def _handle_shutdown(self):
        with self.lifecycle_lock:
            self.stream_clients = 0
            self._stop_stream()

    def _reader_loop(self):
        pending = bytearray()
        try:
            while self.reader_running:
                available = self.serial_conn.in_waiting
                chunk = self.serial_conn.read(min(available, 4096) or 1)
                if not chunk:
                    continue
                pending.extend(chunk)
                while True:
                    delimiter = b'\x00' if self.binary_mode else b'\n'
                    end = pending.find(delimiter)
                    if end < 0:
                        break
                    record = bytes(pending[:end])
                    del pending[:end + 1]
                    host_time = self.reactor.monotonic()
                    if self.binary_mode:
                        if record:
                            self._handle_binary_record(record, host_time)
                    else:
                        line = record.rstrip(b'\r').decode(
                            'utf-8', 'replace')
                        if line:
                            self._handle_ascii_line(line, host_time)
                if len(pending) > SERIAL_RECORD_LIMIT:
                    raise IOError(
                        "Trinkey serial record exceeded %d bytes"
                        % (SERIAL_RECORD_LIMIT,))
        except Exception as e:
            if self.reader_running:
                self.reader_error = e
                logging.exception("Trinkey stream reader failed")
            self.reader_running = False
            with self.control_condition:
                self.control_condition.notify_all()

    def _handle_ascii_line(self, line, host_time):
        for match in ACK_STREAM_STOP_RE.finditer(line):
            with self.control_condition:
                self.ack_stream_stop = match.group(0)
                self.stream_stopped = True
                self.control_condition.notify_all()
        for match in SYNC_RE.finditer(line):
            sequence, device_time_us = match.groups()
            with self.control_condition:
                self.idle_sync_responses[int(sequence)] = (
                    int(device_time_us), host_time)
                self.control_condition.notify_all()
        for match in ACK_STREAM_START_RE.finditer(line):
            with self.control_condition:
                self.ack_stream_start = match.groups()
                self.binary_mode = True
                self.control_condition.notify_all()
        for match in ACK_BW_RE.finditer(line):
            with self.control_condition:
                self.ack_bandwidth = int(match.group(1))
                self.control_condition.notify_all()
        if line.startswith('ERR,'):
            logging.warning("Trinkey firmware response: %s", line)

    def _handle_binary_record(self, encoded, host_time):
        try:
            raw = _cobs_decode(encoded)
            if len(raw) < 11:
                raise ValueError("short stream frame")
            magic, frame_type, payload_length = struct.unpack_from(
                '<IBH', raw, 0)
            if magic != STREAM_MAGIC:
                raise ValueError("bad stream magic")
            if len(raw) != 7 + payload_length + 4:
                raise ValueError("bad stream payload length")
            expected_crc = struct.unpack_from('<I', raw, len(raw) - 4)[0]
            actual_crc = zlib.crc32(raw[:-4]) & 0xffffffff
            if expected_crc != actual_crc:
                raise ValueError("bad stream CRC")
            payload = raw[7:-4]
            if frame_type == STREAM_FRAME_DATA:
                self._handle_data_frame(payload)
            elif frame_type == STREAM_FRAME_SYNC:
                self._handle_sync_frame(payload, host_time)
            elif frame_type == STREAM_FRAME_STOP:
                self._handle_status_frame(payload)
                with self.control_condition:
                    self.stream_stopped = True
                    self.binary_mode = False
                    self.control_condition.notify_all()
            elif frame_type == STREAM_FRAME_STATUS:
                self._handle_status_frame(payload)
            else:
                raise ValueError("unknown stream frame type %d" % frame_type)
        except Exception:
            self.frame_errors += 1
            logging.exception("Invalid Trinkey stream frame")

    def _handle_data_frame(self, payload):
        if len(payload) < STREAM_DATA_HEADER.size:
            raise ValueError("short data frame")
        values = STREAM_DATA_HEADER.unpack_from(payload)
        (packet_sequence, sample_rate, sensor_mask, sample_count,
         total_sample_count, dropped_count, transport_dropped_count,
         overrun_count, max_loop_us, max_base_read_us,
         max_toolhead_read_us) = values
        expected_length = (
            STREAM_DATA_HEADER.size + sample_count * STREAM_SAMPLE.size)
        if len(payload) != expected_length:
            raise ValueError("incorrect data-frame sample count")
        if self.last_packet_sequence is not None:
            expected_sequence = (self.last_packet_sequence + 1) & 0xffffffff
            if packet_sequence != expected_sequence:
                self.packet_sequence_errors += (
                    packet_sequence - expected_sequence) & 0xffffffff
        self.last_packet_sequence = packet_sequence
        self.firmware_status = {
            'sample_rate': sample_rate,
            'sensor_mask': sensor_mask,
            'sample_count': total_sample_count,
            'dropped': dropped_count,
            'transport_dropped': transport_dropped_count,
            'overruns': overrun_count,
            'max_loop_us': max_loop_us,
            'max_base_read_us': max_base_read_us,
            'max_toolhead_read_us': max_toolhead_read_us,
        }

        decoded = []
        offset = STREAM_DATA_HEADER.size
        for unused in range(sample_count):
            decoded.append(STREAM_SAMPLE.unpack_from(payload, offset))
            offset += STREAM_SAMPLE.size

        with self.data_lock:
            for sample in decoded:
                (sequence, base_t_us, base_x, base_y, base_z,
                 tool_t_us, tool_x, tool_y, tool_z, flags) = sample
                if flags & FLAG_BASE_VALID:
                    queue = self.sample_queues['base']
                    if len(queue) == queue.maxlen:
                        self.host_queue_drops += 1
                    queue.append((
                        sequence, base_t_us, base_x, base_y, base_z, flags))
                elif flags & FLAG_BASE_ERROR:
                    self.read_errors += 1
                if flags & FLAG_TOOLHEAD_VALID:
                    queue = self.sample_queues['toolhead']
                    if len(queue) == queue.maxlen:
                        self.host_queue_drops += 1
                    queue.append((
                        sequence, tool_t_us, tool_x, tool_y, tool_z, flags))
                elif flags & FLAG_TOOLHEAD_ERROR:
                    self.read_errors += 1

    def _handle_sync_frame(self, payload, host_after):
        if len(payload) != STREAM_SYNC.size:
            raise ValueError("incorrect sync-frame length")
        sequence, device_time_us = STREAM_SYNC.unpack(payload)
        with self.control_condition:
            host_before = self.stream_sync_pending.pop(sequence, None)
            if host_before is not None:
                self.stream_sync_responses.append(
                    (device_time_us, host_before, host_after))
            self.control_condition.notify_all()

    def _handle_status_frame(self, payload):
        if len(payload) != STREAM_STATUS.size:
            raise ValueError("incorrect status-frame length")
        values = STREAM_STATUS.unpack(payload)
        (sample_rate, batch_size, sensor_mask, sample_count, dropped_count,
         transport_dropped_count, overrun_count, max_loop_us,
         max_base_read_us, max_toolhead_read_us) = values
        self.firmware_status = {
            'sample_rate': sample_rate,
            'batch_size': batch_size,
            'sensor_mask': sensor_mask,
            'sample_count': sample_count,
            'dropped': dropped_count,
            'transport_dropped': transport_dropped_count,
            'overruns': overrun_count,
            'max_loop_us': max_loop_us,
            'max_base_read_us': max_base_read_us,
            'max_toolhead_read_us': max_toolhead_read_us,
        }

    def _sync_timer_event(self, eventtime):
        if not self.streaming or self.serial_conn is None:
            return self.reactor.NEVER
        host_before = self.reactor.monotonic()
        sequence = self.next_sync_sequence
        self.next_sync_sequence += 1
        with self.control_condition:
            self.stream_sync_pending[sequence] = host_before
            stale_before = host_before - SYNC_TIMEOUT
            stale = [key for key, value in self.stream_sync_pending.items()
                     if value < stale_before]
            for key in stale:
                del self.stream_sync_pending[key]
            self.sync_timeouts += len(stale)
        try:
            self._send_command('STREAM_SYNC,%d' % (sequence,))
        except Exception:
            with self.control_condition:
                if self.stream_sync_pending.pop(sequence, None) is not None:
                    self.sync_timeouts += 1
            logging.exception("Unable to send Trinkey stream sync")
        return eventtime + self.sync_interval

    def _update_clock_mapping(self):
        with self.control_condition:
            responses = self.stream_sync_responses
            self.stream_sync_responses = []
        for device_time_us, host_before, host_after in responses:
            host_mid = .5 * (host_before + host_after)
            print_time = self.mcu.estimated_print_time(host_mid)
            self.clock_mapper.add(
                device_time_us, print_time, host_after - host_before)

    def api_update(self, sensor, eventtime):
        if self.reader_error is not None:
            raise self._command_error(
                "Trinkey reader failed: %s" % (self.reader_error,))
        self._update_clock_mapping()
        with self.data_lock:
            samples = list(self.sample_queues[sensor])
            self.sample_queues[sensor].clear()
        if not samples or not self.clock_mapper.ready:
            return {}
        scale = self.raw_scale
        data = []
        for sequence, device_time_us, x_raw, y_raw, z_raw, flags in samples:
            print_time = self.clock_mapper.get_print_time(device_time_us)
            data.append((
                round(print_time, 9), device_time_us, sequence,
                x_raw, y_raw, z_raw,
                round(x_raw * scale, 6),
                round(y_raw * scale, 6),
                round(z_raw * scale, 6), flags))
        clock_status = self.clock_mapper.get_status()
        return {
            'data': data,
            'firmware': dict(self.firmware_status),
            'clock': clock_status,
            'host_queue_drops': self.host_queue_drops,
            'frame_errors': self.frame_errors,
            'packet_sequence_errors': self.packet_sequence_errors,
            'sync_timeouts': self.sync_timeouts,
            'read_errors': self.read_errors,
        }

    def api_update_experiment(self, eventtime):
        """Return bounded sensor batches with a sampled nominal reference."""
        if self.reader_error is not None:
            raise self._command_error(
                "Trinkey reader failed: %s" % (self.reader_error,))
        self._update_clock_mapping()
        if not self.clock_mapper.ready:
            return {}

        # A delayed event loop must not produce one unbounded JSON response.
        # One second per sensor drains a backlog ten times faster than it is
        # created at the default 100 ms API interval, while bounding each
        # TrapQ extraction and socket message.
        sample_batches = {}
        with self.data_lock:
            for sensor in self.sensors:
                queue = self.sample_queues[sensor]
                count = min(len(queue), self.rate)
                sample_batches[sensor] = [
                    queue.popleft() for unused in range(count)]
        if not any(sample_batches.values()):
            return {}

        mapped = {}
        timed_keys = []
        for sensor, samples in sample_batches.items():
            sensor_rows = []
            for row_index, sample in enumerate(samples):
                (sequence, device_time_us, x_raw, y_raw, z_raw,
                 flags) = sample
                print_time = self.clock_mapper.get_print_time(device_time_us)
                sensor_rows.append((
                    print_time, device_time_us, sequence,
                    x_raw, y_raw, z_raw, flags))
                timed_keys.append((print_time, (sensor, row_index)))
            mapped[sensor] = sensor_rows
        timed_keys.sort()

        start_time = timed_keys[0][0]
        end_time = timed_keys[-1][0] + 1.e-9
        motion = self.printer.lookup_object('motion_report')
        toolhead_trapq = motion.trapqs.get('toolhead')
        if toolhead_trapq is None:
            raise self._command_error(
                "Toolhead TrapQ is unavailable for Trinkey reference data")
        moves, _cdata = toolhead_trapq.extract_trapq(start_time, end_time)
        references = _sample_trapq_motion(moves, timed_keys)

        current_print_time = self.mcu.estimated_print_time(eventtime)
        valid_after = current_print_time - TRAPQ_HISTORY_SAFE_AGE
        scale = self.raw_scale
        data = {}
        for sensor, rows in mapped.items():
            output_rows = []
            command_axis = 1 if sensor == 'base' else 0
            axis = 'y' if sensor == 'base' else 'x'
            raw_axis = 1 if sensor == 'base' else 0
            for row_index, row in enumerate(rows):
                (print_time, device_time_us, sequence,
                 x_raw, y_raw, z_raw, flags) = row
                reference_position, reference_acceleration, reference_found = (
                    references[(sensor, row_index)])
                history_valid = print_time >= valid_after
                motion_valid = history_valid and self.kinematics is not None
                commanded_xyz = None
                if motion_valid:
                    try:
                        commanded_xyz = self._get_commanded_position(print_time)
                    except Exception:
                        self.motion_lookup_errors += 1
                        motion_valid = False
                        if self.motion_lookup_errors == 1:
                            logging.exception(
                                "Unable to reconstruct commanded position at "
                                "%.9f (further errors are counted in status)",
                                print_time)

                motor_position = 0.
                motor_velocity = 0.
                observed_position = 0.
                observer_acceleration = 0.
                observer_valid = 0
                if motion_valid:
                    motor_position = commanded_xyz[command_axis]
                    raw_values = (x_raw, y_raw, z_raw)
                    (motor_velocity, observed_position,
                     observer_acceleration, observer_valid) = (
                        self._observe_axis(
                            axis, print_time, motor_position,
                            raw_values[raw_axis] * scale))
                elif axis in self.observer_axes:
                    self._reset_axis_state(self.observer_axes[axis])

                output_rows.append((
                    round(print_time, 9), device_time_us, sequence,
                    x_raw, y_raw, z_raw,
                    round(x_raw * scale, 6),
                    round(y_raw * scale, 6),
                    round(z_raw * scale, 6), flags,
                    round(reference_position[command_axis], 6),
                    round(reference_acceleration[command_axis], 6),
                    round(motor_position, 6), round(motor_velocity, 6),
                    round(observed_position, 6),
                    round(observer_acceleration, 6), observer_valid,
                    int(history_valid and reference_found)))
            data[sensor] = output_rows

        data.update({
            'firmware': dict(self.firmware_status),
            'clock': self.clock_mapper.get_status(),
            'host_queue_drops': self.host_queue_drops,
            'frame_errors': self.frame_errors,
            'packet_sequence_errors': self.packet_sequence_errors,
            'sync_timeouts': self.sync_timeouts,
            'read_errors': self.read_errors,
            'observer_resets': self.observer_resets,
            'motion_lookup_errors': self.motion_lookup_errors,
        })
        return data

    def start_internal_client(self, sensor):
        if sensor not in self.sensor_dumps:
            raise self._command_error(
                "Unknown Trinkey accelerometer '%s'" % (sensor,))
        return self.sensor_dumps[sensor].start_internal_client()

    def start_observer_client(self):
        """Subscribe to the shared experiment/observer batches."""
        return self.experiment_dump.start_internal_client()

    def is_streaming(self):
        return self.streaming or self.stream_clients > 0

    cmd_TRINKEY_ACCEL_STATUS_help = (
        "Report Trinkey streaming, timing, and loss diagnostics")
    def cmd_TRINKEY_ACCEL_STATUS(self, gcmd):
        status = self.get_status(self.reactor.monotonic())
        keys = (
            'streaming', 'rate', 'batch_size', 'bandwidth',
            'clock_ready', 'clock_points', 'clock_accepted_points',
            'clock_rtt_minimum', 'clock_rtt_limit',
            'clock_rtt_tolerance', 'clock_slope', 'clock_last_rtt',
            'sync_timeouts', 'reader_alive', 'reader_error',
            'firmware_sample_count',
            'firmware_dropped', 'firmware_transport_dropped',
            'firmware_overruns', 'firmware_max_loop_us',
            'firmware_max_base_read_us',
            'firmware_max_toolhead_read_us', 'host_queue_drops',
            'packet_sequence_errors', 'frame_errors', 'read_errors',
            'observer_enabled', 'observer_resets', 'motion_lookup_errors',
            'observer_x_bias_ready', 'observer_y_bias_ready')
        fields = [
            '%s=%s' % (key, status[key])
            for key in keys if key in status]
        gcmd.respond_info('Trinkey accel: ' + ' '.join(fields))

    def get_status(self, eventtime):
        clock_status = self.clock_mapper.get_status()
        status = {
            'connected': self.serial_conn is not None,
            'streaming': self.streaming,
            'rate': self.rate,
            'batch_size': self.batch_size,
            'bandwidth': self.bandwidth,
            'sensors': self.sensors,
            'clock_ready': clock_status['ready'],
            'clock_points': clock_status['points'],
            'clock_accepted_points': clock_status['accepted_points'],
            'clock_rtt_minimum': clock_status['rtt_minimum'],
            'clock_rtt_limit': clock_status['rtt_limit'],
            'clock_rtt_tolerance': clock_status['rtt_tolerance'],
            'clock_slope': clock_status['slope'],
            'clock_offset': clock_status['offset'],
            'clock_last_rtt': clock_status['last_rtt'],
            'clock_last_device_time_us':
                clock_status['last_device_time_us'],
            'clock_last_print_time': clock_status['last_print_time'],
            'sync_timeouts': self.sync_timeouts,
            'reader_alive': bool(
                self.reader_thread is not None
                and self.reader_thread.is_alive()),
            'reader_error': ('' if self.reader_error is None
                             else str(self.reader_error)),
            'host_queue_drops': self.host_queue_drops,
            'frame_errors': self.frame_errors,
            'packet_sequence_errors': self.packet_sequence_errors,
            'read_errors': self.read_errors,
            'observer_enabled': bool(self.observer_axes),
            'observer_resets': self.observer_resets,
            'motion_lookup_errors': self.motion_lookup_errors,
        }
        for axis, observer in self.observer_axes.items():
            status['observer_%s_bias_ready' % (axis,)] = (
                observer['bias_ready'])
            status['observer_%s_bias' % (axis,)] = observer['bias']
        status.update(
            ('firmware_' + key, value)
            for key, value in self.firmware_status.items())
        return status


def load_config(config):
    return TrinkeyAccel(config)
