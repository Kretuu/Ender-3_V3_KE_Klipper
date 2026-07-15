# Sinusoidal vibration test support
#
# Copyright (C) 2026 Jakub Kreczetowski <kret1315@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import copy, csv, json, logging, math, os, threading, time

try:
    import serial
except ImportError:
    serial = None


AXIS_INDEX = {'X': 0, 'Y': 1, 'Z': 2}

DEFAULT_AMPLITUDE = 0.02
DEFAULT_DURATION = 5.0
DEFAULT_FREQUENCY = 40.0
DEFAULT_SEGMENTS_PER_CYCLE = 32
DEFAULT_MAX_SEGMENT_RATE = 6000.0
DEFAULT_CHUNK_TIME = 0.250
DEFAULT_RAMP_CYCLES = 2.0
DEFAULT_MAX_SEGMENTS = 100000
DEFAULT_TRINKEY_BAUDRATE = 115200
DEFAULT_TRINKEY_LOG_DIR = "~/printer_data/logs/vibration_tests"
DEFAULT_TRINKEY_SYNC_COUNT = 20
DEFAULT_TRINKEY_SYNC_TIMEOUT = 1.0
TRINKEY_SYNC_INTERVAL = 0.010

TRINKEY_SAMPLE_HEADER = (
    'chip_id', 'sample', 't_us', 'ax_raw', 'ay_raw', 'az_raw')
TRINKEY_SYNC_HEADER = (
    'phase', 'seq', 'host_before_s', 'host_after_s', 'host_mid_s',
    'print_time_mid_s', 'trinkey_t_us', 'rtt_s')
TRINKEY_INPUT_HEADER = (
    'segment', 'axis', 'axis_index', 'frequency_hz', 'print_time_s',
    'relative_t_s', 'dt_s', 'offset_mm', 'start_velocity_mm_s',
    'accel_mm_s2', 'end_offset_mm')
TRINKEY_EVENT_HEADER = (
    'event', 'host_time_s', 'print_time_s', 'trinkey_t_us', 'line')


class TrinkeyLogger:
    """Log Trinkey samples and synchronisation data for one vibration test."""

    def __init__(self, printer, port, log_dir, run_id):
        """Create a logger bound to one USB serial port and output directory.

        Args:
            printer: Klipper printer object used for reactor and MCU time.
            port: USB CDC serial device path for the Trinkey.
            log_dir: Parent directory where run directories are created.
            run_id: Short directory-safe identifier for the current run.
        """
        self.printer = printer
        self.reactor = printer.get_reactor()
        self.mcu = printer.lookup_object('mcu')
        self.port = port
        self.log_dir = os.path.expanduser(log_dir)
        self.run_id = self._sanitize_run_id(run_id)
        self.run_dir = os.path.join(self.log_dir, self.run_id)

        self.serial_conn = None
        self.reader_thread = None
        self.running = False
        self.started = False
        self.reader_error = None
        self.next_sync_seq = 1
        self.sync_responses = {}
        self.sync_condition = threading.Condition()
        self.file_lock = threading.Lock()

        self.sample_file = None
        self.sync_file = None
        self.input_file = None
        self.event_file = None
        self.sample_writer = None
        self.sync_writer = None
        self.input_writer = None
        self.event_writer = None

    def _sanitize_run_id(self, run_id):
        """Return a run identifier that is safe to use as a directory name.

        Args:
            run_id: User or caller supplied run identifier.

        Returns:
            Directory-safe identifier containing only simple ASCII characters.
        """
        cleaned = []
        for ch in str(run_id or ""):
            if ch.isalnum() or ch in ('-', '_', '.'):
                cleaned.append(ch)
            else:
                cleaned.append('_')
        out = ''.join(cleaned).strip('._')
        if not out:
            out = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return out[:96]

    def start(self, metadata):
        """Open serial/files, start the reader thread, and send START.

        Args:
            metadata: JSON-serialisable dictionary describing the test point.
        """
        if serial is None:
            raise self.printer.command_error(
                "TRINKEY=1 requires the pyserial Python package")

        os.makedirs(self.run_dir)
        try:
            self._open_files()
            self._write_metadata(metadata)
            self.serial_conn = serial.Serial(
                self.port, DEFAULT_TRINKEY_BAUDRATE, timeout=0.050,
                write_timeout=0.500)
            self.serial_conn.reset_input_buffer()
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            self._send_command("START,%s" % (self.run_id,))
            self.started = True
            self.write_event("start_sent")
        except Exception:
            self.stop()
            raise

    def stop(self):
        """Stop firmware streaming, stop the reader thread, and close files."""
        if self.serial_conn is not None and self.started:
            try:
                self._send_command("STOP")
                self._pause(0.100)
            except Exception:
                logging.exception("Failed to send Trinkey STOP")

        self.running = False
        if self.reader_thread is not None:
            self.reader_thread.join(0.500)
            self.reader_thread = None

        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                logging.exception("Failed to close Trinkey serial port")
            self.serial_conn = None

        self._close_files()
        self.started = False

    def _open_files(self):
        """Create CSV files and write their headers."""
        self.sample_file = open(os.path.join(self.run_dir, "samples.csv"),
                                "w", newline="")
        self.sync_file = open(os.path.join(self.run_dir, "sync.csv"),
                              "w", newline="")
        self.input_file = open(os.path.join(self.run_dir, "input.csv"),
                               "w", newline="")
        self.event_file = open(os.path.join(self.run_dir, "events.csv"),
                               "w", newline="")

        self.sample_writer = csv.DictWriter(
            self.sample_file, fieldnames=TRINKEY_SAMPLE_HEADER)
        self.sync_writer = csv.DictWriter(
            self.sync_file, fieldnames=TRINKEY_SYNC_HEADER)
        self.input_writer = csv.DictWriter(
            self.input_file, fieldnames=TRINKEY_INPUT_HEADER)
        self.event_writer = csv.DictWriter(
            self.event_file, fieldnames=TRINKEY_EVENT_HEADER)

        self.sample_writer.writeheader()
        self.sync_writer.writeheader()
        self.input_writer.writeheader()
        self.event_writer.writeheader()

    def _close_files(self):
        """Flush and close all files that were opened for the run."""
        for fh in (self.sample_file, self.sync_file,
                   self.input_file, self.event_file):
            if fh is None:
                continue
            try:
                fh.flush()
                fh.close()
            except Exception:
                logging.exception("Failed to close Trinkey log file")
        self.sample_file = self.sync_file = None
        self.input_file = self.event_file = None
        self.sample_writer = self.sync_writer = None
        self.input_writer = self.event_writer = None

    def _write_metadata(self, metadata):
        """Write the run metadata JSON beside the CSV files.

        Args:
            metadata: JSON-serialisable dictionary describing the test point.
        """
        metadata = dict(metadata)
        metadata['run_id'] = self.run_id
        metadata['trinkey_port'] = self.port
        metadata['run_dir'] = self.run_dir
        metadata['files'] = {
            'samples': 'samples.csv',
            'sync': 'sync.csv',
            'input': 'input.csv',
            'events': 'events.csv',
        }
        filename = os.path.join(self.run_dir, "metadata.json")
        with open(filename, "w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _send_command(self, command):
        """Send one newline-terminated command to the Trinkey firmware.

        Args:
            command: Command without the trailing newline.
        """
        if self.serial_conn is None:
            raise self.printer.command_error("Trinkey serial port is not open")
        self.serial_conn.write(("%s\n" % (command,)).encode("ascii"))
        self.serial_conn.flush()

    def _reader_loop(self):
        """Continuously read Trinkey serial rows and route them to CSV files."""
        try:
            while self.running:
                raw_line = self.serial_conn.readline()
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                host_time = self.reactor.monotonic()
                self._handle_serial_line(line, host_time)
        except Exception as e:
            self.reader_error = e
            logging.exception("Trinkey reader thread failed")
            with self.sync_condition:
                self.sync_condition.notify_all()

    def _handle_serial_line(self, line, host_time):
        """Parse one firmware line and write it to the appropriate CSV.

        Args:
            line: Decoded line without newline characters.
            host_time: Klipper host monotonic time when the line was received.
        """
        if line.startswith("A,"):
            self._handle_sample_line(line)
            return
        if line.startswith("SYNC,"):
            self._handle_sync_line(line, host_time)
            return
        if line.startswith("ACK_START,"):
            self.write_event("ack_start", host_time=host_time,
                             trinkey_t_us=self._last_csv_int(line))
            return
        if line.startswith("ACK_STOP,"):
            self.write_event("ack_stop", host_time=host_time,
                             trinkey_t_us=self._last_csv_int(line))
            return
        if line.startswith("ERR,"):
            self.write_event("firmware_error", host_time=host_time, line=line)
            return
        self.write_event("raw", host_time=host_time, line=line)

    def _handle_sample_line(self, line):
        """Parse and store one tagged accelerometer sample row.

        Args:
            line: Firmware row in A,chip_id,sample,t_us,ax_raw,ay_raw,az_raw
                format.
        """
        parts = line.split(',')
        if len(parts) != 7:
            self.write_event("bad_sample", line=line)
            return
        row = {
            'chip_id': parts[1],
            'sample': parts[2],
            't_us': parts[3],
            'ax_raw': parts[4],
            'ay_raw': parts[5],
            'az_raw': parts[6],
        }
        with self.file_lock:
            self.sample_writer.writerow(row)

    def _handle_sync_line(self, line, host_time):
        """Store one firmware sync reply for the waiting sync request.

        Args:
            line: Firmware row in SYNC,sequence,trinkey_t_us format.
            host_time: Klipper host monotonic receive time.
        """
        parts = line.split(',')
        if len(parts) != 3:
            self.write_event("bad_sync", host_time=host_time, line=line)
            return
        try:
            seq = int(parts[1])
            trinkey_t_us = int(parts[2])
        except ValueError:
            self.write_event("bad_sync", host_time=host_time, line=line)
            return
        with self.sync_condition:
            self.sync_responses[seq] = (trinkey_t_us, host_time)
            self.sync_condition.notify_all()

    def _last_csv_int(self, line):
        """Return the final comma-separated field as int if possible.

        Args:
            line: Comma-separated firmware row.

        Returns:
            Integer value, or an empty string when parsing fails.
        """
        try:
            return int(line.rsplit(',', 1)[1])
        except (IndexError, ValueError):
            return ""

    def sync_many(self, phase, count):
        """Run several sync requests and write their timing records.

        Args:
            phase: Short label such as pre or post.
            count: Number of sync requests to perform.
        """
        for _ in range(count):
            self.sync_once(phase)
            self._pause(TRINKEY_SYNC_INTERVAL)

    def sync_once(self, phase):
        """Send one SYNC request and log the clock correspondence row.

        Args:
            phase: Short label identifying when the sync was taken.
        """
        seq = self.next_sync_seq
        self.next_sync_seq += 1

        host_before = self.reactor.monotonic()
        self._send_command("SYNC,%d" % (seq,))
        trinkey_t_us, host_after = self._wait_for_sync_response(seq)
        host_mid = 0.5 * (host_before + host_after)
        print_time_mid = self.mcu.estimated_print_time(host_mid)

        row = {
            'phase': phase,
            'seq': seq,
            'host_before_s': self._fmt(host_before),
            'host_after_s': self._fmt(host_after),
            'host_mid_s': self._fmt(host_mid),
            'print_time_mid_s': self._fmt(print_time_mid),
            'trinkey_t_us': trinkey_t_us,
            'rtt_s': self._fmt(host_after - host_before),
        }
        with self.file_lock:
            self.sync_writer.writerow(row)
            self.sync_file.flush()

    def _wait_for_sync_response(self, seq):
        """Wait until the reader thread receives a matching SYNC reply.

        Args:
            seq: Sequence number that identifies the sync request.

        Returns:
            Tuple of Trinkey timestamp in microseconds and host receive time.
        """
        deadline = self.reactor.monotonic() + DEFAULT_TRINKEY_SYNC_TIMEOUT
        with self.sync_condition:
            while seq not in self.sync_responses:
                if self.reader_error is not None:
                    raise self.printer.command_error(
                        "Trinkey reader failed: %s" % (self.reader_error,))
                remaining = deadline - self.reactor.monotonic()
                if remaining <= 0.:
                    raise self.printer.command_error(
                        "Timed out waiting for Trinkey SYNC %d" % (seq,))
                self.sync_condition.wait(remaining)
            return self.sync_responses.pop(seq)

    def write_input_segments(self, axis, axis_index, frequency, segments):
        """Write the exact motion segments queued into Klipper trapq.

        Args:
            axis: Commanded printer axis as X, Y, or Z.
            axis_index: Klipper numeric axis index.
            frequency: Test frequency in Hz.
            segments: Iterable of segment tuples returned by vibration motion.
        """
        with self.file_lock:
            for segment in segments:
                index, print_time, relative_t, dt, x0, start_v, accel, x1 = (
                    segment)
                self.input_writer.writerow({
                    'segment': index,
                    'axis': axis,
                    'axis_index': axis_index,
                    'frequency_hz': self._fmt(frequency),
                    'print_time_s': self._fmt(print_time),
                    'relative_t_s': self._fmt(relative_t),
                    'dt_s': self._fmt(dt),
                    'offset_mm': self._fmt(x0),
                    'start_velocity_mm_s': self._fmt(start_v),
                    'accel_mm_s2': self._fmt(accel),
                    'end_offset_mm': self._fmt(x1),
                })
            self.input_file.flush()

    def write_event(self, event, host_time=None, print_time=None,
                    trinkey_t_us="", line=""):
        """Write one run event row.

        Args:
            event: Short event name.
            host_time: Optional Klipper host monotonic time.
            print_time: Optional Klipper print time.
            trinkey_t_us: Optional Trinkey timestamp.
            line: Optional raw firmware line.
        """
        if host_time is None:
            host_time = self.reactor.monotonic()
        row = {
            'event': event,
            'host_time_s': self._fmt(host_time),
            'print_time_s': "" if print_time is None else self._fmt(print_time),
            'trinkey_t_us': trinkey_t_us,
            'line': line,
        }
        with self.file_lock:
            if self.event_writer is not None:
                self.event_writer.writerow(row)

    def _pause(self, seconds):
        """Yield to Klipper's reactor for a short host-side delay.

        Args:
            seconds: Delay duration in seconds.
        """
        self.reactor.pause(self.reactor.monotonic() + seconds)

    def _fmt(self, value):
        """Format floating-point times and motion values consistently.

        Args:
            value: Numeric value to serialise into CSV.

        Returns:
            String with enough precision for timing and FRF processing.
        """
        return "%.9f" % (value,)


class ValidationMove:
    def __init__(self, printer, max_accel, start_pos, end_pos, speed):
        """Build a minimal move-like object for kinematics validation.

        Args:
            printer: Klipper printer object used to create command errors.
            max_accel: Maximum acceleration available for the validation move.
            start_pos: Current toolhead position.
            end_pos: Candidate end position to validate.
            speed: Maximum speed used for the validation move.
        """
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
        """Apply kinematics-imposed velocity and acceleration limits.

        Args:
            speed: Maximum velocity allowed by the kinematics check.
            accel: Maximum acceleration allowed by the kinematics check.
        """
        self.max_cruise_v2 = min(self.max_cruise_v2, speed * speed)
        self.accel = min(self.accel, accel)

    def move_error(self, msg="Move out of range"):
        """Create a Klipper command error for an invalid validation move.

        Args:
            msg: Error message prefix.

        Returns:
            Klipper command error object.
        """
        ep = self.end_pos
        return self.printer.command_error(
            "%s: %.3f %.3f %.3f [%.3f]"
            % (msg, ep[0], ep[1], ep[2], ep[3]))


class VibrationTest:
    def __init__(self, config):
        """Register the vibration test G-code command and read config defaults.

        Args:
            config: Klipper config section for this extra.
        """
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.reactor = self.printer.get_reactor()
        self.trinkey_enabled = config.getboolean('trinkey', False)
        self.trinkey_port = config.get('trinkey_port', None)
        self.trinkey_log_dir = config.get(
            'trinkey_log_dir', DEFAULT_TRINKEY_LOG_DIR)
        self.trinkey_sync_count = config.getint(
            'trinkey_sync_count', DEFAULT_TRINKEY_SYNC_COUNT, minval=0,
            maxval=200)
        self.gcode.register_command(
            "RUN_VIBRATION_TEST", self.cmd_RUN_VIBRATION_TEST,
            desc=self.cmd_RUN_VIBRATION_TEST_help)

    def _default_run_id(self, axis, frequency):
        """Build a simple timestamped run identifier for Trinkey logging.

        Args:
            axis: Commanded printer axis.
            frequency: Test frequency in Hz.

        Returns:
            Directory-safe run identifier.
        """
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return "%s_%s_%.3fhz" % (timestamp, axis.lower(), frequency)

    def _create_trinkey_logger(self, gcmd, axis, frequency):
        """Create the Trinkey logger requested by a RUN_VIBRATION_TEST command.

        Args:
            gcmd: Klipper G-code command wrapper.
            axis: Commanded printer axis.
            frequency: Test frequency in Hz.

        Returns:
            Tuple of TrinkeyLogger and sync count.
        """
        port = gcmd.get('TRINKEY_PORT', self.trinkey_port)
        if not port:
            raise gcmd.error(
                "TRINKEY=1 requires TRINKEY_PORT=<usb serial path> or "
                "trinkey_port in [vibration_test]")
        log_dir = gcmd.get('LOG_DIR', self.trinkey_log_dir)
        run_id = gcmd.get('RUN_ID', self._default_run_id(axis, frequency))
        sync_count = gcmd.get_int(
            'SYNC_COUNT', self.trinkey_sync_count, minval=0, maxval=200)
        return TrinkeyLogger(self.printer, port, log_dir, run_id), sync_count

    def _build_trinkey_metadata(
            self, axis, axis_index, duration, frequency, amplitude, ramp_time,
            segment_time, segments_per_cycle, max_segment_rate, chunk_time,
            max_segments, max_v, max_a):
        """Build JSON metadata written beside the Trinkey CSV files.

        Args:
            axis: Commanded printer axis.
            axis_index: Klipper numeric axis index.
            duration: Commanded test duration in seconds.
            frequency: Commanded test frequency in Hz.
            amplitude: Commanded displacement amplitude in mm.
            ramp_time: Ramp-in/ramp-out duration in seconds.
            segment_time: Duration of each generated trapq segment in seconds.
            segments_per_cycle: Requested segment resolution per sine cycle.
            max_segment_rate: User limit for generated segments per second.
            chunk_time: Host-side trapq flush chunk duration.
            max_segments: User limit for total generated segments.
            max_v: Maximum commanded velocity in mm/s.
            max_a: Maximum commanded acceleration in mm/s^2.

        Returns:
            Dictionary suitable for metadata.json.
        """
        return {
            'protocol': 'trinkey_bno055_sync_v1',
            'axis': axis,
            'axis_index': axis_index,
            'duration_s': duration,
            'frequency_hz': frequency,
            'amplitude_mm': amplitude,
            'ramp_time_s': ramp_time,
            'segment_time_s': segment_time,
            'segments_per_cycle': segments_per_cycle,
            'max_segment_rate': max_segment_rate,
            'chunk_time_s': chunk_time,
            'max_segments': max_segments,
            'peak_velocity_mm_s': max_v,
            'peak_accel_mm_s2': max_a,
            'sample_units': {
                'accel_raw': 'BNO055 raw acceleration LSB',
                'trinkey_t_us': 'Trinkey time_us_64 microseconds',
                'print_time_s': 'Klipper primary MCU print_time seconds',
            },
        }

    def _get_axis(self, gcmd):
        """Read and validate the requested printer axis.

        Args:
            gcmd: Klipper G-code command wrapper.

        Returns:
            Tuple of axis name and numeric axis index.
        """
        axis = gcmd.get('AXIS', default='X').upper()
        if axis not in AXIS_INDEX:
            raise gcmd.error(
                '{"code":"key274", "msg": "Invalid axis: %s", '
                '"values": ["%s"]}' % (axis, axis))
        return axis, AXIS_INDEX[axis]

    def _get_ramp_time(self, gcmd, duration, frequency):
        """Determine the ramp-in/ramp-out duration for the sinusoid.

        Args:
            gcmd: Klipper G-code command wrapper.
            duration: Total test duration in seconds.
            frequency: Sinusoid frequency in Hz.

        Returns:
            Ramp duration in seconds.
        """
        ramp_time = gcmd.get_float('RAMP_TIME', None, minval=0.,
                                   maxval=duration * 0.5)
        if ramp_time is not None:
            return ramp_time
        ramp_cycles = gcmd.get_float('RAMP_CYCLES', DEFAULT_RAMP_CYCLES,
                                     minval=0.)
        return min(duration * 0.25, ramp_cycles / frequency)

    def _envelope(self, t, duration, ramp_time):
        """Calculate the smooth amplitude envelope at time t.

        Args:
            t: Time from the start of the vibration command in seconds.
            duration: Total test duration in seconds.
            ramp_time: Ramp-in/ramp-out duration in seconds.

        Returns:
            Envelope multiplier between 0 and 1.
        """
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
        """Calculate commanded displacement offset for the sine test.

        Args:
            t: Time from the start of the vibration command in seconds.
            duration: Total test duration in seconds.
            frequency: Sinusoid frequency in Hz.
            amplitude: Peak displacement amplitude in mm.
            ramp_time: Ramp-in/ramp-out duration in seconds.

        Returns:
            Commanded axis offset in mm.
        """
        omega = 2. * math.pi * frequency
        return (amplitude * self._envelope(t, duration, ramp_time)
                * math.sin(omega * t))

    def _iter_segments(self, duration, frequency, amplitude, ramp_time,
                       segment_time):
        """Generate constant-acceleration segments approximating the sine wave.

        Args:
            duration: Total test duration in seconds.
            frequency: Sinusoid frequency in Hz.
            amplitude: Peak displacement amplitude in mm.
            ramp_time: Ramp-in/ramp-out duration in seconds.
            segment_time: Requested segment duration in seconds.

        Yields:
            Tuples of relative start time, duration, start offset,
            start velocity, constant acceleration, and end offset.
        """
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
        """Scan generated segments to calculate count and peak requirements.

        Args:
            duration: Total test duration in seconds.
            frequency: Sinusoid frequency in Hz.
            amplitude: Peak displacement amplitude in mm.
            ramp_time: Ramp-in/ramp-out duration in seconds.
            segment_time: Requested segment duration in seconds.

        Returns:
            Tuple containing segment count, min offset, max offset,
            peak velocity, and peak acceleration.
        """
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
        """Validate that the requested vibration motion is physically allowed.

        Args:
            gcmd: Klipper G-code command wrapper.
            toolhead: Klipper toolhead object.
            axis_index: Numeric axis index.
            start_pos: Current toolhead position.
            min_x: Minimum commanded offset in mm.
            max_x: Maximum commanded offset in mm.
            max_v: Peak commanded velocity in mm/s.
            max_a: Peak commanded acceleration in mm/s^2.
        """
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
        """Copy mutable kinematics state before validation-only checks.

        Args:
            kin: Klipper kinematics object.

        Returns:
            Dictionary of copied attributes that must be restored later.
        """
        saved = {}
        for name in ('limit_xy2', 'limit_z', 'limits',
                     'need_home', 'homed_axis'):
            if hasattr(kin, name):
                saved[name] = copy.deepcopy(getattr(kin, name))
        return saved

    def _restore_kinematics_state(self, kin, saved):
        """Restore kinematics state changed by validation-only checks.

        Args:
            kin: Klipper kinematics object.
            saved: State dictionary returned by _save_kinematics_state().
        """
        for name, value in saved.items():
            setattr(kin, name, value)

    def _enter_direct_motion(self, toolhead):
        """Prepare the toolhead for directly appending trapq motion segments.

        Args:
            toolhead: Klipper toolhead object.

        Returns:
            Safe future Klipper print_time for the first vibration segment.
        """
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
        """Append one vibration segment directly to the toolhead trapq.

        Args:
            toolhead: Klipper toolhead object.
            print_time: Segment start time in Klipper print-time seconds.
            start_pos: Toolhead position at the start of the test.
            axis_index: Numeric axis index to excite.
            offset: Segment start offset from start_pos in mm.
            dt: Segment duration in seconds.
            start_v: Segment start velocity in mm/s.
            accel: Segment constant acceleration in mm/s^2.
        """
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
        """Update Klipper's commanded toolhead position after direct trapq use.

        Args:
            toolhead: Klipper toolhead object.
            start_pos: Toolhead position at the start of the test.
            axis_index: Numeric axis index that was excited.
            offset: Final commanded offset from start_pos in mm.
        """
        pos = list(start_pos)
        if abs(offset) < 0.000000001:
            offset = 0.
        pos[axis_index] += offset
        toolhead.commanded_pos[:] = pos

    def _run_direct_sinusoid(self, toolhead, axis_index, duration, frequency,
                             amplitude, ramp_time, segment_time, chunk_time):
        """Queue the sinusoidal vibration motion directly into Klipper trapq.

        Args:
            toolhead: Klipper toolhead object.
            axis_index: Numeric axis index to excite.
            duration: Total test duration in seconds.
            frequency: Sinusoid frequency in Hz.
            amplitude: Peak displacement amplitude in mm.
            ramp_time: Ramp-in/ramp-out duration in seconds.
            segment_time: Requested segment duration in seconds.
            chunk_time: Print-time interval between step-generation flushes.

        Returns:
            Tuple containing logged input segments, motion start print_time,
            and motion end print_time.
        """
        start_pos = toolhead.get_position()
        print_time = self._enter_direct_motion(toolhead)
        start_print_time = print_time
        chunk_start = print_time
        input_segments = []

        for index, segment in enumerate(self._iter_segments(
                duration, frequency, amplitude, ramp_time, segment_time)):
            t0, dt, x0, start_v, accel, x1 = segment
            input_segments.append(
                (index, print_time, t0, dt, x0, start_v, accel, x1))
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
        return input_segments, start_print_time, print_time

    cmd_RUN_VIBRATION_TEST_help = (
        "Run streamed sinusoidal motion. Params: AXIS=<X|Y|Z> "
        "DURATION=<0.1..10s> FREQUENCY=<Hz> AMPLITUDE=<mm> "
        "SEGMENTS_PER_CYCLE=<8..200> MAX_SEGMENT_RATE=<segments/s> "
        "CHUNK_TIME=<0.05..1s> MAX_SEGMENTS=<count> RAMP_TIME=<s> "
        "RAMP_CYCLES=<cycles> MAX_VELOCITY=<mm/s> "
        "MAX_ACCEL=<mm/s^2> INPUT_SHAPING=<0|1> WAIT=<0|1> "
        "TRINKEY=<0|1> TRINKEY_PORT=<path> LOG_DIR=<path> RUN_ID=<id> "
        "SYNC_COUNT=<count>")

    def cmd_RUN_VIBRATION_TEST(self, gcmd):
        """Execute the RUN_VIBRATION_TEST G-code command.

        Args:
            gcmd: Klipper G-code command wrapper.
        """
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
        trinkey_enabled = gcmd.get_int(
            'TRINKEY', int(self.trinkey_enabled), minval=0, maxval=1)
        if trinkey_enabled and not wait:
            raise gcmd.error("TRINKEY=1 requires WAIT=1 so the log covers "
                             "the complete motion")

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

        trinkey_logger = None
        try:
            if trinkey_enabled:
                trinkey_logger, sync_count = self._create_trinkey_logger(
                    gcmd, axis, frequency)
                trinkey_metadata = self._build_trinkey_metadata(
                    axis, axis_index, duration, frequency, amplitude,
                    ramp_time, segment_time, segments_per_cycle,
                    max_segment_rate, chunk_time, max_segments, max_v, max_a)
                trinkey_logger.start(trinkey_metadata)
                trinkey_logger.write_event("pre_sync_start")
                trinkey_logger.sync_many("pre", sync_count)

            gcmd.respond_info(
                "Vibration test %s: %.3f Hz, %.6f mm, %.3f s, "
                "%d segments (%.0f/s), peak %.3f mm/s, %.3f mm/s^2"
                % (axis, frequency, amplitude, duration, count, segment_rate,
                   max_v, max_a))
            input_segments, motion_start, motion_end = (
                self._run_direct_sinusoid(
                    toolhead, axis_index, duration, frequency, amplitude,
                    ramp_time, segment_time, chunk_time))
            if trinkey_logger is not None:
                trinkey_logger.write_event("motion_queued",
                                           print_time=motion_start)
                trinkey_logger.write_event("motion_queue_end",
                                           print_time=motion_end)
            if wait:
                toolhead.wait_moves()
            if trinkey_logger is not None:
                trinkey_logger.write_event("motion_wait_complete")
                trinkey_logger.write_event("post_sync_start")
                trinkey_logger.sync_many("post", sync_count)
                trinkey_logger.write_input_segments(
                    axis, axis_index, frequency, input_segments)
                gcmd.respond_info(
                    "Trinkey vibration log written to %s"
                    % (trinkey_logger.run_dir,))
        finally:
            if trinkey_logger is not None:
                trinkey_logger.stop()
            if input_shaper is not None:
                input_shaper.enable_shaping()
                gcmd.respond_info("Re-enabled [input_shaper]")


def load_config(config):
    """Create the vibration test extra from a Klipper config section.

    Args:
        config: Klipper config section for this extra.

    Returns:
        VibrationTest instance registered with Klipper.
    """
    return VibrationTest(config)
