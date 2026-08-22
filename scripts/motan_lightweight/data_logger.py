#!/usr/bin/env python
# Lightweight Motan logger for the dissertation acceleration experiments
#
# Copyright (C) 2020-2021  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import sys, os, optparse, socket, select, json, errno, time, zlib

INDEX_UPDATE_TIME = 5.0
TRINKEY_DATA_WARNING = 5.0
ClientInfo = {'program': 'motan_lightweight_logger', 'version': 'v0.3'}

# Only subscribe to status needed to identify a run, align its print time, and
# diagnose a failed acquisition.  The combined experiment stream is subscribed
# separately below.
STATUS_OBJECTS = {
    'webhooks': ['state', 'state_message'],
    'configfile': ['settings'],
    'print_stats': [
        'filename', 'state', 'message', 'total_duration',
        'print_duration', 'filament_used'],
    'virtual_sdcard': [
        'file_path', 'progress', 'is_active', 'file_position', 'file_size'],
    'filtered_bspline': [
        'enabled', 'mode', 'hybrid_observation_errors',
        'hybrid_worker_queue', 'hybrid_worker_queue_max',
        'hybrid_worker_drops', 'hybrid_worker_alive',
        'hybrid_x', 'hybrid_y'],
    'toolhead': ['print_time', 'estimated_print_time', 'stalls'],
    'system_stats': ['sysload', 'cputime', 'memavail'],
    'trinkey_accel': None,
}
REQUIRED_TRINKEY_SENSORS = ('base', 'toolhead')

def webhook_socket_create(uds_filename):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.setblocking(0)
    sys.stderr.write("Waiting for connect to %s\n" % (uds_filename,))
    while 1:
        try:
            sock.connect(uds_filename)
        except socket.error as e:
            if e.errno == errno.ECONNREFUSED:
                time.sleep(0.1)
                continue
            sys.stderr.write("Unable to connect socket %s [%d,%s]\n"
                             % (uds_filename, e.errno,
                                errno.errorcode[e.errno]))
            sys.exit(-1)
        break
    sys.stderr.write("Connection.\n")
    return sock

class LogWriter:
    def __init__(self, filename):
        self.file = open(filename, "wb")
        self.comp = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION,
                                     zlib.DEFLATED, 31)
        self.raw_pos = self.file_pos = 0
    def add_data(self, data):
        d = self.comp.compress(data + b"\x03")
        self.file.write(d)
        self.file_pos += len(d)
        self.raw_pos += len(data) + 1
    def flush(self, flag=zlib.Z_FULL_FLUSH):
        if not self.raw_pos:
            return self.file_pos
        d = self.comp.flush(flag)
        self.file.write(d)
        self.file_pos += len(d)
        return self.file_pos
    def close(self):
        self.flush(zlib.Z_FINISH)
        self.file.close()
        self.file = None
        self.comp = None

class DataLogger:
    def __init__(self, uds_filename, log_prefix):
        # IO
        self.webhook_socket = webhook_socket_create(uds_filename)
        self.poll = select.poll()
        self.poll.register(self.webhook_socket, select.POLLIN | select.POLLHUP)
        self.socket_data = b""
        # Data log
        self.logger = LogWriter(log_prefix + ".json.gz")
        self.index = LogWriter(log_prefix + ".index.gz")
        # Handlers
        self.query_handlers = {}
        self.async_handlers = {}
        # get_status databasing
        self.db = {}
        self.next_index_time = 0.
        self.trinkey_data_deadlines = {}
        self.trinkey_data_warned = set()
        # Start login process
        self.send_query("info", "info", {"client_info": ClientInfo},
                        self.handle_info)
    def error(self, msg):
        sys.stderr.write(msg + "\n")
    def finish(self, msg, status=0):
        self.error(msg)
        # Notify Klipper that the streaming client is gone before potentially
        # slow gzip finalization, so its STREAM_STOP handshake starts promptly.
        try:
            self.poll.unregister(self.webhook_socket)
        except Exception:
            pass
        try:
            self.webhook_socket.close()
        except Exception:
            pass
        self.logger.close()
        self.index.close()
        sys.exit(status)
    # Unix Domain Socket IO
    def send_query(self, msg_id, method, params, cb):
        self.query_handlers[msg_id] = cb
        msg = {"id": msg_id, "method": method, "params": params}
        cm = json.dumps(msg, separators=(',', ':')).encode()
        self.webhook_socket.send(cm + b"\x03")
    def process_socket(self):
        data = self.webhook_socket.recv(4096)
        if not data:
            self.finish("ERROR: Klipper socket closed", status=1)
        parts = data.split(b"\x03")
        parts[0] = self.socket_data + parts[0]
        self.socket_data = parts.pop()
        for part in parts:
            try:
                msg = json.loads(part)
            except:
                self.error("ERROR: Unable to parse line")
                continue
            self.logger.add_data(part)
            msg_q = msg.get("q")
            if msg_q is not None:
                hdl = self.async_handlers.get(msg_q)
                if hdl is not None:
                    hdl(msg, part)
                continue
            msg_id = msg.get("id")
            hdl = self.query_handlers.get(msg_id)
            if hdl is not None:
                del self.query_handlers[msg_id]
                hdl(msg, part)
                if not self.query_handlers:
                    self.flush_index()
                continue
            self.error("ERROR: Message with unknown id")
    def run(self):
        try:
            while 1:
                res = self.poll.poll(1000.)
                for fd, event in res:
                    if fd == self.webhook_socket.fileno():
                        self.process_socket()
                self.check_trinkey_data()
        except KeyboardInterrupt as e:
            self.finish("Keyboard Interrupt")
    # Query response handlers
    def send_subscribe(self, msg_id, method, params, cb=None, async_cb=None):
        if cb is None:
            cb = self.handle_dump
        if async_cb is not None:
            self.async_handlers[msg_id] = async_cb
        params["response_template"] = {"q": msg_id}
        self.send_query(msg_id, method, params, cb)
    def handle_info(self, msg, raw_msg):
        if msg["result"]["state"] != "ready":
            self.finish("ERROR: Klipper not in ready state", status=1)
        self.send_subscribe(
            "status", "objects/subscribe", {"objects": STATUS_OBJECTS},
            self.handle_subscribe, self.handle_async_db)
    def handle_subscribe(self, msg, raw_msg):
        if "result" not in msg:
            self.finish(
                "ERROR: Unable to subscribe to lightweight status: %s"
                % (msg.get("error", {}).get("message", ""),), status=1)
        result = msg["result"]
        self.next_index_time = result["eventtime"] + INDEX_UPDATE_TIME
        self.db["status"] = status = result["status"]
        config = status.get("configfile", {}).get("settings", {})
        trinkey_config = config.get("trinkey_accel")
        if trinkey_config is None:
            self.finish(
                "ERROR: [trinkey_accel] is missing from Klipper config",
                status=1)
        sensors = trinkey_config.get("sensors", REQUIRED_TRINKEY_SENSORS)
        if isinstance(sensors, str):
            sensors = sensors.replace(',', ' ').split()
        sensors = {str(sensor).strip().lower() for sensor in sensors}
        missing = [sensor for sensor in REQUIRED_TRINKEY_SENSORS
                   if sensor not in sensors]
        if missing:
            self.finish(
                "ERROR: Missing required Trinkey sensor(s): %s"
                % (", ".join(missing),), status=1)

        # This combined endpoint samples nominal motion and reconstructs final
        # step commands at each genuine accelerometer timestamp.  It avoids
        # exporting full high-density queues through the Klipper webhook.
        self.send_subscribe(
            "trinkey_accel:experiment", "trinkey_accel/dump_experiment", {},
            async_cb=self.handle_trinkey_dump)
    def handle_dump(self, msg, raw_msg):
        msg_id = msg["id"]
        if "result" not in msg:
            self.finish(
                "ERROR: Unable to subscribe to '%s': %s"
                % (msg_id, msg.get("error", {}).get("message", "")),
                status=1)
        self.db.setdefault("subscriptions", {})[msg_id] = msg["result"]
        if msg_id == "trinkey_accel:experiment":
            deadline = time.monotonic() + TRINKEY_DATA_WARNING
            self.trinkey_data_deadlines.update(
                (sensor, deadline) for sensor in REQUIRED_TRINKEY_SENSORS)
    def handle_trinkey_dump(self, msg, raw_msg):
        params = msg.get("params", {})
        for sensor in REQUIRED_TRINKEY_SENSORS:
            if not params.get(sensor):
                continue
            if sensor in self.trinkey_data_warned:
                self.error("WARNING: Data resumed from trinkey_accel:%s"
                           % (sensor,))
                self.trinkey_data_warned.remove(sensor)
            self.trinkey_data_deadlines[sensor] = (
                time.monotonic() + TRINKEY_DATA_WARNING)
    def check_trinkey_data(self):
        now = time.monotonic()
        stale = sorted(name for name, deadline
                       in self.trinkey_data_deadlines.items()
                       if now >= deadline)
        for sensor in stale:
            if sensor in self.trinkey_data_warned:
                continue
            self.trinkey_data_warned.add(sensor)
            self.error(
                "WARNING: No data from trinkey_accel:%s for %.1f seconds; "
                "capture remains active"
                % (sensor, TRINKEY_DATA_WARNING))
    def flush_index(self):
        self.db['file_position'] = self.logger.flush()
        self.index.add_data(json.dumps(self.db, separators=(',', ':')).encode())
        self.db = {"status": {}}
    def handle_async_db(self, msg, raw_msg):
        params = msg["params"]
        db_status = self.db['status']
        for k, v in params.get("status", {}).items():
            db_status.setdefault(k, {}).update(v)
        trinkey = params.get("status", {}).get("trinkey_accel", {})
        if trinkey.get("reader_error"):
            self.finish(
                "ERROR: Trinkey stream failed: %s"
                % (trinkey["reader_error"],), status=1)
        eventtime = params['eventtime']
        if eventtime >= self.next_index_time:
            self.next_index_time = eventtime + INDEX_UPDATE_TIME
            self.flush_index()

def nice():
    try:
        # Try to re-nice writing process
        os.nice(10)
    except:
        pass

def main():
    usage = "%prog [options] <socket filename> <log name>"
    opts = optparse.OptionParser(usage)
    options, args = opts.parse_args()
    if len(args) != 2:
        opts.error("Incorrect number of arguments")

    nice()
    dl = DataLogger(args[0], args[1])
    dl.run()

if __name__ == '__main__':
    main()
