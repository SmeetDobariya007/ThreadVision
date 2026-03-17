# inspector.py — ThreadVision AI ISO Tolerance Checker + GPIO Output
# ===================================================================
# Compares measured thread dimensions against ISO 262 tolerances
# loaded from config.py. Controls PASS/FAIL LEDs via GPIO.
#
# Usage:
#   from inspector import check, trigger_gpio
#   overall, per_dim = check(measurements)
#   trigger_gpio(overall)

import time
import logging
import threading

from config import (
    TOLERANCES, NOMINAL_VALUES,
    PIN_LED_PASS, PIN_LED_FAIL, PIN_BACKLIGHT,
    LED_ON_DURATION_MS, IS_RASPBERRY_PI,
)

logger = logging.getLogger("ThreadVision.inspector")

# ---------------------------------------------------------------------------
# Try to import RPi.GPIO (only available on Raspberry Pi)
# ---------------------------------------------------------------------------
_GPIO_AVAILABLE = False
_gpio = None
try:
    import RPi.GPIO as GPIO
    _gpio = GPIO
    _GPIO_AVAILABLE = True
    logger.info("RPi.GPIO found — hardware LED control enabled.")
except ImportError:
    logger.info("RPi.GPIO not found — using simulated GPIO output.")


# Dimension display names for human-readable output
DIM_NAMES = {
    'major_diameter': 'Major Diameter',
    'minor_diameter': 'Minor Diameter',
    'pitch':          'Pitch',
    'thread_depth':   'Thread Depth',
    'flank_angle':    'Flank Angle',
}

DIM_UNITS = {
    'major_diameter': 'mm',
    'minor_diameter': 'mm',
    'pitch':          'mm',
    'thread_depth':   'mm',
    'flank_angle':    '°',
}


class InspectionResult:
    """
    Structured result from a thread inspection.

    Attributes:
        overall_pass (bool): True if ALL dimensions pass, False if ANY fails.
        per_dimension (dict): Per-dimension results, each containing:
            - 'value': float or None — measured value
            - 'nominal': float — expected nominal value
            - 'tolerance': (float, float) — (lower, upper) bounds
            - 'status': str — 'PASS', 'FAIL', or 'UNMEASURED'
            - 'deviation': float or None — difference from nominal
        summary (str): Human-readable summary string.
    """

    def __init__(self, overall_pass, per_dimension, summary=""):
        self.overall_pass = overall_pass
        self.per_dimension = per_dimension
        self.summary = summary

    def __repr__(self):
        status = "PASS" if self.overall_pass else "FAIL"
        return f"InspectionResult({status}, {len(self.per_dimension)} dims)"

    def to_dict(self):
        """Convert to a flat dictionary for logging/serialization."""
        result = {
            'overall': 'PASS' if self.overall_pass else 'FAIL',
        }
        for dim, info in self.per_dimension.items():
            result[f"{dim}_value"] = info['value']
            result[f"{dim}_status"] = info['status']
            result[f"{dim}_deviation"] = info['deviation']
        return result


def check(measurements):
    """
    Check measured thread dimensions against ISO tolerances.

    Compares each of the 5 dimensions against the tolerance ranges
    specified in config.py. A dimension passes if its measured value
    falls within [lower_bound, upper_bound]. If any dimension is
    None (unmeasured), it is flagged as 'UNMEASURED' but does NOT
    automatically cause a FAIL — only out-of-tolerance values fail.

    Args:
        measurements (dict): Output from measurement.measure().
            Expected keys: 'major_diameter', 'minor_diameter',
            'pitch', 'thread_depth', 'flank_angle'.
            Values may be float or None.

    Returns:
        tuple: (InspectionResult, str) where:
            - InspectionResult contains all per-dimension analysis.
            - str is 'PASS', 'FAIL', or 'UNMEASURED' (if no dims measured).
    """
    per_dimension = {}
    any_fail = False
    any_measured = False

    for dim_key in ['major_diameter', 'minor_diameter', 'pitch',
                     'thread_depth', 'flank_angle']:
        value = measurements.get(dim_key)
        nominal = NOMINAL_VALUES.get(dim_key)
        tolerance = TOLERANCES.get(dim_key)

        if tolerance is None:
            logger.warning(f"No tolerance defined for '{dim_key}' — skipping.")
            continue

        lower, upper = tolerance

        if value is None:
            status = 'UNMEASURED'
            deviation = None
            logger.info(f"  {DIM_NAMES[dim_key]:16s} : UNMEASURED")
        else:
            any_measured = True
            deviation = value - nominal if nominal else None

            if lower <= value <= upper:
                status = 'PASS'
                symbol = "✓"
            else:
                status = 'FAIL'
                any_fail = True
                symbol = "✗"

            unit = DIM_UNITS[dim_key]
            logger.info(
                f"  {symbol} {DIM_NAMES[dim_key]:16s} : "
                f"{value:7.3f} {unit} "
                f"[{lower:.3f} – {upper:.3f}] "
                f"{'→ ' + status}"
            )

        per_dimension[dim_key] = {
            'value': value,
            'nominal': nominal,
            'tolerance': (lower, upper),
            'status': status,
            'deviation': deviation,
            'name': DIM_NAMES.get(dim_key, dim_key),
            'unit': DIM_UNITS.get(dim_key, ''),
        }

    # Determine overall result
    if not any_measured:
        overall_pass = False
        overall_str = 'UNMEASURED'
        summary = "No dimensions could be measured. Check bolt placement and focus."
    elif any_fail:
        overall_pass = False
        overall_str = 'FAIL'
        failed_dims = [
            DIM_NAMES[k] for k, v in per_dimension.items()
            if v['status'] == 'FAIL'
        ]
        summary = f"FAIL — {', '.join(failed_dims)} out of tolerance."
    else:
        overall_pass = True
        overall_str = 'PASS'
        summary = "All measured dimensions within ISO tolerance."

    logger.info(f"  Overall: {overall_str} — {summary}")

    result = InspectionResult(overall_pass, per_dimension, summary)
    return result, overall_str


# =========================================================================
# GPIO Control
# =========================================================================

def setup_gpio():
    """
    Initialize GPIO pins for LED output.

    Call this once at application startup. On non-Pi platforms,
    this is a no-op.
    """
    if not _GPIO_AVAILABLE:
        logger.info("GPIO setup skipped (not on Raspberry Pi).")
        return

    try:
        _gpio.setmode(_gpio.BCM)
        _gpio.setwarnings(False)

        # Configure output pins
        _gpio.setup(PIN_LED_PASS, _gpio.OUT, initial=_gpio.LOW)
        _gpio.setup(PIN_LED_FAIL, _gpio.OUT, initial=_gpio.LOW)
        _gpio.setup(PIN_BACKLIGHT, _gpio.OUT, initial=_gpio.LOW)

        logger.info(
            f"GPIO initialized: PASS=GPIO{PIN_LED_PASS}, "
            f"FAIL=GPIO{PIN_LED_FAIL}, BACKLIGHT=GPIO{PIN_BACKLIGHT}"
        )
    except Exception as e:
        logger.error(f"GPIO setup failed: {e}")


def trigger_gpio(overall_pass):
    """
    Activate the appropriate PASS or FAIL LED.

    Turns on the green (PASS) or red (FAIL) LED for LED_ON_DURATION_MS,
    then turns both off. Runs in a background thread so it doesn't
    block the GUI.

    Args:
        overall_pass (bool): True for PASS (green), False for FAIL (red).
    """
    if not _GPIO_AVAILABLE:
        # Simulate on non-Pi platforms
        color = "GREEN (PASS)" if overall_pass else "RED (FAIL)"
        logger.info(f"GPIO simulated: {color} LED ON for {LED_ON_DURATION_MS}ms")
        print(f"  [SIM] LED: {color}")
        return

    def _led_sequence():
        try:
            # Turn off both
            _gpio.output(PIN_LED_PASS, _gpio.LOW)
            _gpio.output(PIN_LED_FAIL, _gpio.LOW)

            # Turn on the appropriate LED
            if overall_pass:
                _gpio.output(PIN_LED_PASS, _gpio.HIGH)
            else:
                _gpio.output(PIN_LED_FAIL, _gpio.HIGH)

            # Hold for duration
            time.sleep(LED_ON_DURATION_MS / 1000.0)

            # Turn off
            _gpio.output(PIN_LED_PASS, _gpio.LOW)
            _gpio.output(PIN_LED_FAIL, _gpio.LOW)
        except Exception as e:
            logger.error(f"GPIO LED sequence error: {e}")

    thread = threading.Thread(target=_led_sequence, daemon=True)
    thread.start()


def set_backlight(on):
    """
    Control the LED backlight panel.

    Args:
        on (bool): True to turn backlight ON, False for OFF.
    """
    if not _GPIO_AVAILABLE:
        state = "ON" if on else "OFF"
        logger.info(f"Backlight simulated: {state}")
        return

    try:
        _gpio.output(PIN_BACKLIGHT, _gpio.HIGH if on else _gpio.LOW)
        logger.info(f"Backlight {'ON' if on else 'OFF'}")
    except Exception as e:
        logger.error(f"Backlight control failed: {e}")


def cleanup_gpio():
    """
    Clean up GPIO resources. Call at application shutdown.
    """
    if not _GPIO_AVAILABLE:
        return

    try:
        _gpio.output(PIN_LED_PASS, _gpio.LOW)
        _gpio.output(PIN_LED_FAIL, _gpio.LOW)
        _gpio.output(PIN_BACKLIGHT, _gpio.LOW)
        _gpio.cleanup()
        logger.info("GPIO cleaned up.")
    except Exception as e:
        logger.warning(f"GPIO cleanup error: {e}")


# ---------------------------------------------------------------------------
# Self-test: run `python inspector.py` to verify tolerance checking
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ThreadVision AI — Inspector Self-Test")
    print("=" * 60)

    # Test 1: All dimensions PASS
    print("\n--- Test 1: Good bolt (all PASS) ---")
    good_bolt = {
        'major_diameter': 7.995,
        'minor_diameter': 6.550,
        'pitch':          1.248,
        'thread_depth':   0.650,
        'flank_angle':    60.2,
    }
    result, status = check(good_bolt)
    assert status == 'PASS', f"Expected PASS, got {status}"
    print(f"  Result: {status} ✓")

    # Test 2: Major diameter out of tolerance
    print("\n--- Test 2: Bad major diameter (FAIL) ---")
    bad_bolt = {
        'major_diameter': 7.900,   # Below 7.972 lower bound
        'minor_diameter': 6.550,
        'pitch':          1.248,
        'thread_depth':   0.650,
        'flank_angle':    60.2,
    }
    result, status = check(bad_bolt)
    assert status == 'FAIL', f"Expected FAIL, got {status}"
    assert result.per_dimension['major_diameter']['status'] == 'FAIL'
    print(f"  Result: {status} ✓")

    # Test 3: Unmeasured pitch
    print("\n--- Test 3: Unmeasured pitch ---")
    partial = {
        'major_diameter': 7.990,
        'minor_diameter': 6.550,
        'pitch':          None,
        'thread_depth':   0.650,
        'flank_angle':    60.0,
    }
    result, status = check(partial)
    assert result.per_dimension['pitch']['status'] == 'UNMEASURED'
    print(f"  Result: {status} (pitch unmeasured) ✓")

    # Test GPIO simulation
    print("\n--- Test: GPIO simulation ---")
    trigger_gpio(True)
    trigger_gpio(False)

    print("\n✓ Inspector self-test passed.")
