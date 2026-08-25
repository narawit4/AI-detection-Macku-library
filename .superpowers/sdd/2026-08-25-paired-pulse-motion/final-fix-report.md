# Paired-Pulse Motion Final Fix Report

## Status

Complete. All final-review findings were addressed in the isolated
`feature/paired-pulse-motion` worktree. No Nuitka build was run.

## Fixes

- `motion.py` now treats `OverflowError` from `float(...)` like other invalid
  numeric input and falls back to the field default.
- `ConfigStore.load()` now catches plain `ValueError` from `json.load()`,
  including Python's integer digit-limit failure, and returns safe defaults
  with an actionable warning.
- Schema identifiers are accepted only when they are exact integral values.
  Booleans, fractional numbers, fractional strings, and malformed strings are
  rejected without truncation. Ambiguous identifiers disable saving so the
  source document cannot be overwritten; the existing schema `> 2` protection
  remains intact.
- `stop_motion()` now sets an event snapshot before taking any service lock.
  A dedicated short cancellation-bookkeeping lock protects generation/reason
  state, and the existing movement lock remains the return barrier that
  guarantees no move begins after `stop_motion()` returns. The current event
  and generation are revalidated after the lock-free signal to cover stale
  event races.
- High-risk tests now assert the exact early Smooth sequence, the exact 25 ms
  half-pulse wait at 20 Hz, deterministic blocked-move STOP behavior, and
  frozen `MotionSettings` immutability.
- Fresh defaults and schema-1 migration label the exact default motion values
  as `Balanced`. The UI starts with `Balanced` selected while Jitter remains
  `DISABLED`.

## TDD Evidence

### Numeric conversion and JSON parsing

RED:

```text
python -m unittest discover -s tests -p test_motion.py -k huge_integer_values -v
ERROR: OverflowError: int too large to convert to float

python -m unittest discover -s tests -p test_settings.py -k oversized_json_integer -v
ERROR: ValueError: Exceeds the limit (640 digits) for integer string conversion
```

GREEN after adding the two narrow exception handlers:

```text
test_huge_integer_values_use_safe_defaults_instead_of_overflowing ... ok
test_oversized_json_integer_uses_defaults_and_reports_warning ... ok
```

The complete suite then passed 235 tests.

### Exact schema identifiers and overwrite protection

RED:

```text
test_ambiguous_schema_identifiers_disable_saving
  schema=True        FAIL: True is not false
  schema=1.5         FAIL: True is not false
  schema='1.5'       FAIL: True is not false
  schema='malformed' FAIL: True is not false

test_fractional_newer_schema_is_not_truncated_or_overwritten
  FAIL: True is not false
```

GREEN after exact integral validation and conservative save disabling:

```text
python -m unittest discover -s tests -p test_settings.py -k schema -v
Ran 7 tests ... OK
```

This included the unchanged future-schema regression. The complete suite then
passed 237 tests.

### Motion test strengthening

The restored immutability test passed against the existing frozen dataclass:

```text
test_motion_settings_snapshot_is_immutable ... ok
```

The exact Smooth sequence was mutation-tested by temporarily removing
`magnitude_residual` from the pair magnitude calculation.

RED under that mutation:

```text
test_smooth_ramp_has_exact_early_fractional_residual_sequence ... FAIL
First differing element 4: (0, 0) != (0, -1)
```

GREEN after restoring residual carry:

```text
test_smooth_ramp_has_exact_early_fractional_residual_sequence ... ok
```

The half-pulse timing test was mutation-tested by temporarily using the full
pair interval.

RED under that mutation:

```text
test_worker_waits_one_half_pulse_interval ... FAIL
[0.05] != [0.025]
```

GREEN with the required half-pulse interval restored:

```text
test_worker_waits_one_half_pulse_interval ... ok
```

### Immediate STOP and move serialization

RED with cancellation still inside the movement lock:

```text
test_stop_signals_while_move_is_blocked_and_serializes_its_return ... FAIL
AssertionError: False is not true
```

The failed assertion was the cancellation event wait while the gated
controller deliberately held `controller.move()` blocked.

GREEN after two-phase cancellation:

```text
test_stop_signals_while_move_is_blocked_and_serializes_its_return ... ok
python -m unittest discover -s tests -p test_makcu_service.py -v
Ran 21 tests ... OK
```

The deterministic test also verifies that STOP has not returned while the
in-flight move remains blocked, the requested `gated_stop` reason is retained,
and the move-start count cannot increase after `stop_motion()` returns.

### Balanced default and migration label

RED:

```text
test_missing_config_returns_safe_disabled_defaults ... FAIL
'Custom' != 'Balanced'

test_schema_one_preserves_app_choices_but_migrates_motion_to_defaults ... FAIL
'Custom' != 'Balanced'

test_fresh_balanced_preset_still_starts_disabled ... FAIL
'Custom' != 'Balanced'
```

GREEN after updating the default and schema-1 migration label:

```text
test_missing_config_returns_safe_disabled_defaults ... ok
test_schema_one_preserves_app_choices_but_migrates_motion_to_defaults ... ok
test_fresh_balanced_preset_still_starts_disabled ... ok
```

The complete suite then passed 241 tests.

## Final Verification

Run fresh from
`C:\Users\User\Desktop\Jitter\.worktrees\paired-pulse-motion`:

```text
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
Exit 0

python -m unittest discover -s tests -v
Ran 241 tests in 24.141s
OK

python -c "import makcu"
Exit 0

git diff --check
Exit 0
```

`git diff --check` emitted only Git's configured LF-to-CRLF conversion notices;
it reported no whitespace errors.

## Hardware Status

Not performed. No claim is made for physical Makcu connection, button gating,
movement direction/rate, reconnect, Test 3s, hotkey, STOP, disconnect, or
shutdown behavior on hardware. Hardware-free fake-controller tests passed.

## Concerns

- Physical Makcu behavior remains pending connected-device verification.
- Nuitka/package verification was intentionally not run.
