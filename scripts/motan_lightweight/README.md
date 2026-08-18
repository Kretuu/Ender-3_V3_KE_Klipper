# Lightweight Motan logger

This is the supported Motan variant for the dissertation acceleration prints.
The stock logger remains generic and does not subscribe to the Trinkey stream.

The lightweight logger records only:

- raw base and toolhead Trinkey samples;
- nominal TrapQ position and acceleration;
- final motor position reconstructed from Klipper's step history; and
- the shared state observer's position estimate and corrected acceleration;
- a small status subset for run identification, print-time alignment, and
  acquisition diagnostics; and
- the filtered B-spline mode and X/Y hybrid training, solve, delay, weight-norm,
  and fallback status.

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
  /usr/data/printer_data/logs/motan/sine35_comp_r1
```

Stop it with `Ctrl-C` after the print. It writes the standard Motan pair:
`sine35_comp_r1.json.gz` and `sine35_comp_r1.index.gz`. The copied
`readlog.py`, `analyzers.py`, and `motan_graph.py` understand the same format.

For hybrid captures, use an unambiguous prefix such as
`sine35_hybrid_r1`. The status stream records whether the learner passed its
warm-up and whether hybrid preview solves actually occurred; the filename alone
is not proof that hybrid compensation was active.

The principal datasets are:

```text
trinkey_accel(toolhead,desired_position)
trinkey_accel(toolhead,motor_position)
trinkey_accel(toolhead,observed_position)
trinkey_accel(base,desired_position)
trinkey_accel(base,motor_position)
trinkey_accel(base,observed_position)
```

The `desired_*` values come from nominal toolhead TrapQ. The `motor_*` values
come from the final generated step history, so they include any active FBF or
input-shaper transformation and step quantisation. The `observed_*` position is
the model-assisted estimate corrected by measured acceleration.

`observer_valid` is zero during the initial 0.4-second stationary bias
measurement and on a sample following a timing discontinuity. `reference_valid`
is zero if a delayed sample is outside the conservative usable-history limit or
has no matching TrapQ move. Keep the printer stationary for at least 0.4 s
after starting the logger.

Use a unique log prefix for every run. The logger opens output files in write
mode and will replace files that already have the same prefix.
