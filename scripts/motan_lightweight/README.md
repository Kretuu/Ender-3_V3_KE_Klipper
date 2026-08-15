# Lightweight Motan logger

This is a separate Motan variant for the dissertation acceleration prints. The
stock logger remains in `scripts/motan` and is unchanged.

The lightweight logger records only:

- base and toolhead Trinkey samples;
- nominal toolhead acceleration evaluated at each genuine sensor timestamp;
- a small status subset for run identification, print-time alignment, and
  acquisition diagnostics.

It uses the custom `trinkey_accel/dump_experiment` endpoint and deliberately
does not subscribe to the full toolhead or extruder TrapQ, step queues, ADXL345
streams, angle sensors, or every Klipper status object. The endpoint performs
one bounded TrapQ extraction and chronological pass for a batch of sensor
timestamps, instead of serialising every high-density motion segment.

If either sensor produces no API data for five seconds, the logger prints a
warning but remains connected. This prevents an event-loop delay from being
misdiagnosed as sensor failure and avoids resetting a Klipper socket while it
still has output queued. Validate the timestamp sequence and all loss counters
after every experiment.

Run it before starting the print:

```sh
python3 scripts/motan_lightweight/data_logger.py \
  /usr/data/printer_data/comms/klippy.sock \
  /usr/data/printer_data/logs/motan/sine_35_comp_r1
```

Stop it with `Ctrl-C` after the print. It writes the standard Motan pair:
`sine_35_comp_r1.json.gz` and `sine_35_comp_r1.index.gz`. The copied
`readlog.py`, `analyzers.py`, and `motan_graph.py` understand the same format.

The principal datasets are:

```text
trinkey_accel(toolhead,x)
trinkey_accel(toolhead,command_x)
trinkey_accel(base,y)
trinkey_accel(base,command_y)
```

`command_x` and `command_y` are the nominal toolhead TrapQ accelerations at the
timestamp of the corresponding physical sample. `reference_valid` is zero if
a delayed sample is older than the conservative 25-second usable-history
limit. Existing lightweight logs containing `trapq(toolhead,...)` remain
readable.

Do not run stock Motan and this lightweight logger at the same time. They are
separate capture modes backed by the same Trinkey stream.

Use a unique log prefix for every run. The logger opens output files in write
mode and will replace files that already have the same prefix.
