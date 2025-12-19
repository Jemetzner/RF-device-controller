#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
import time
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyvisa
import dwfpy as dwf


# -------------------- Signal Generator --------------------
class SMC100A:
    """Simple R&S SMC100A wrapper using SCPI over VISA."""

    def __init__(self, resource: str):
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = 10_000
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"
        try:
            idn = self.inst.query("*IDN?")
        except Exception:
            idn = "<IDN? failed>"
        print("Connected to:", idn)

    def set_freq(self, hz) -> None:
        """Set RF frequency.

        Accepts either a numeric frequency in Hz (float/int) or a raw SCPI
        frequency string such as \"10000 kHz\".
        """
        if isinstance(hz, (int, float)):
            cmd = f"{hz}"
        else:
            cmd = str(hz)
        self.inst.write(f"SOUR:FREQ {cmd}")

    def set_power(self, dbm: float | str) -> None:
        """Set RF power in dBm (float or already-formatted string)."""
        cmd = f"{dbm}" if isinstance(dbm, (int, float)) else str(dbm)
        self.inst.write(f"SOUR:POW:POW {cmd}")

    def rf_on(self) -> None:
        self.inst.write("OUTP ON")

    def rf_off(self) -> None:
        self.inst.write("OUTP OFF")

    def close(self) -> None:
        try:
            self.rf_off()
        finally:
            self.inst.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# -------------------- Oscilloscope --------------------
class DigilentScope:
    """Digilent scope wrapper (via dwfpy) with automatic timebase helpers.

    Public API uses 1-based channel numbering (CH1 = 1, CH2 = 2, ...).
    """

    def __init__(
        self,
        channels: Sequence[int] | None = None,
        sample_rate: float = 20e6,
        buffer_size: int = 8192,
    ):
        if channels is None:
            channels = [1]
        # store 0-based internally
        self.channels = [ch - 1 for ch in channels]

        self.device = dwf.Device()
        self.device.open()
        self.ai = self.device.analog_input

        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Basic per-channel configuration
        for idx in self.channels:
            ch = self.ai[idx]
            ch.enabled = True
            ch.range = 5.0
            ch.offset = 0.0

    def configure_channel(self, channel: int = 1, v_range: float = 5.0, offset: float = 0.0) -> None:
        """Configure a single channel (1-based index)."""
        idx = channel - 1
        ch = self.ai[idx]
        ch.enabled = True
        ch.range = v_range
        ch.offset = offset

    def set_timebase(self, sample_rate: float, buffer_size: int) -> None:
        """Update the scope timebase settings used for subsequent acquisitions."""
        self.sample_rate = float(sample_rate)
        self.buffer_size = int(buffer_size)

    def _wait_for_completion(self, timeout: float = 2.0) -> None:
        t0 = time.time()
        while True:
            status = self.ai.read_status(read_data=True)
            # 0: done, 2: prefill (dwf constants), treat both as finished
            if status in (0, 2):
                break
            if time.time() - t0 > timeout:
                raise TimeoutError("Acquisition did not complete in time")
            time.sleep(0.001)

    def get_waveform(self, channel: int = 1, timeout: float = 2.0) -> np.ndarray:
        """Acquire a single waveform from a channel (1-based index)."""
        idx = channel - 1
        # Use 0-based index for trigger channel in dwfpy
        self.ai.setup_edge_trigger(
            mode="auto",
            channel=idx,
            slope="rising",
            level=0.1,
            hysteresis=0.01,
        )
        self.ai.single(
            configure=True,
            start=True,
            buffer_size=self.buffer_size,
            sample_rate=self.sample_rate,
        )
        self._wait_for_completion(timeout=timeout)
        samples = np.array(self.ai[idx].get_data())
        return samples

    def get_peak_to_peak(self, channel: int = 1, timeout: float = 2.0) -> float:
        """Return Vpp for the specified channel."""
        data = self.get_waveform(channel, timeout=timeout)
        return float(np.max(data) - np.min(data))

    def close(self) -> None:
        self.device.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


VALID_MANTISSAS = np.array([1, 2, 5])


def quantize_sample_rate(fs_ideal: float, fs_min: float = 0.1, fs_max: float = 100e6) -> float:
    """Snap fs_ideal to nearest valid Digilent sample rate (1,2,5 × 10^n)."""
    fs_ideal = np.clip(fs_ideal, fs_min, fs_max)

    decade = np.floor(np.log10(fs_ideal))
    candidates: list[float] = []

    for d in [decade - 1, decade, decade + 1]:
        for m in VALID_MANTISSAS:
            candidates.append(m * 10**d)

    candidates_arr = np.array(candidates)
    candidates_arr = candidates_arr[(candidates_arr >= fs_min) & (candidates_arr <= fs_max)]

    return float(candidates_arr[np.argmin(np.abs(candidates_arr - fs_ideal))])


VALID_BUFFERS = np.array([2**n for n in range(5, 15)])  # 32 → 16384


def select_buffer_size(fs: float, f_signal: float, n_cycles: int = 10) -> int:
    """Pick the smallest power-of-two buffer that captures at least n_cycles."""
    if f_signal <= 0:
        raise ValueError("Signal frequency must be > 0")

    required = fs * (n_cycles / f_signal)

    for buf in VALID_BUFFERS:
        if buf >= required:
            return int(buf)

    return int(VALID_BUFFERS[-1])  # clamp to max


def choose_timebase(f_signal: float, n_cycles: int = 10) -> tuple[float, int]:
    """Return (sample_rate, buffer_size) respecting Digilent constraints.

    We aim for ~5000 samples over n_cycles of the signal and then quantize
    to the allowed (1,2,5)×10^n sample rates and power-of-two buffer sizes.
    """
    if f_signal <= 0:
        raise ValueError("Signal frequency must be > 0")

    # Ideal continuous math: show n_cycles on the screen with ~5000 samples
    display_time = n_cycles / f_signal
    fs_ideal = 5000.0 / display_time

    fs = quantize_sample_rate(fs_ideal)
    buf = select_buffer_size(fs, f_signal, n_cycles)

    return fs, buf



# -------------------- Sweep Script --------------------
def run_sweep(
    sg_resource: str,
    f_start: float,
    f_stop: float,
    n_points: int = 50,
    channel: int = 1,
    output_file: str = "sweep.csv",
    n_cycles: int = 10,
) -> None:
    """Log-spaced frequency sweep with automatic timebase selection.

    For each frequency, this:
      - sets the SMC100A frequency
      - chooses (sample_rate, buffer_size) to capture ~n_cycles
      - measures Vpp on the given Digilent scope channel
      - saves and plots the result
    """
    frequencies = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
    vpp_results: list[float] = []

    with SMC100A(sg_resource) as sg, DigilentScope(channels=[channel]) as scope:
        scope.configure_channel(channel=channel, v_range=5.0, offset=0.0)

        sg.rf_on()
        try:
            for f in frequencies:
                sg.set_freq(f)

                # Timebase: capture ~n_cycles periods on ~5000 points
                fs, buf = choose_timebase(f, n_cycles=n_cycles)
                scope.set_timebase(sample_rate=fs, buffer_size=buf)

                # Small delay to let instruments settle
                time.sleep(0.01)

                # Acquire Vpp
                try:
                    vpp = scope.get_peak_to_peak(channel=channel, timeout=2.0)
                except Exception as e:  # noqa: BLE001
                    print(f"Error at {f/1e6:.3f} MHz: {e}")
                    vpp = float("nan")
                vpp_results.append(vpp)
                print(f"Freq: {f/1e6:.3f} MHz -> Vpp: {vpp:.3f} V")
        finally:
            sg.rf_off()

    # Save results
    data = np.column_stack((frequencies, vpp_results))
    np.savetxt(output_file, data, delimiter=",", header="Frequency(Hz),Vpp(V)", comments="")
    print(f"Data saved to {output_file}")

    # Plot
    plt.figure()
    plt.semilogx(frequencies, vpp_results, marker="o")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Vpp [V]")
    plt.title("Frequency Sweep")
    plt.grid(True, which="both")
    plt.show()


# -------------------- Command-line interface --------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Frequency sweep using SMC100A + Digilent scope")
    parser.add_argument("--sg", type=str, required=True, help="Signal generator VISA resource string")
    parser.add_argument("--fstart", type=float, default=9e3, help="Start frequency (Hz)")
    parser.add_argument("--fstop", type=float, default=100e6, help="Stop frequency (Hz)")
    parser.add_argument("--npoints", type=int, default=50, help="Number of points in sweep")
    parser.add_argument("--channel", type=int, default=1, help="Scope channel to measure")
    parser.add_argument("--outfile", type=str, default="sweep.csv", help="CSV file to save results")
    parser.add_argument("--ncycles", type=int, default=10, help="Number of signal cycles to capture per point")
    args = parser.parse_args()

    run_sweep(
        args.sg,
        args.fstart,
        args.fstop,
        n_points=args.npoints,
        channel=args.channel,
        output_file=args.outfile,
        n_cycles=args.ncycles,
    )