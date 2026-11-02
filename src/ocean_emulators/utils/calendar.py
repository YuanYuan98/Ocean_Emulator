"""Calendar conditioning using the actual timestamps, including CF calendars."""

import numpy as np


def calendar_phase(times) -> np.ndarray:
    """Return fractional years in [0, 1), with January 1 at phase zero.

    Preserve leap years, non-Gregorian calendars and fractional days. This uses
    input timestamps, never label timestamps or a nominal fixed sampling cadence.
    """
    values = np.asarray(times)
    phases = []
    for value in values.flat:
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                raise ValueError("Calendar conditioning cannot use NaT.")
            start = value.astype("datetime64[Y]")
            end = start + np.timedelta64(1, "Y")
            phase = (value - start) / (end.astype("datetime64[D]") - start)
        else:
            start = value.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=start.year + 1)
            phase = (value - start).total_seconds() / (end - start).total_seconds()
        phases.append(float(phase))
    return np.asarray(phases, dtype=np.float32).reshape(values.shape)
