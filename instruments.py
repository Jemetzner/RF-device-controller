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

    def __init__(self, resource: str, visa_backend: str = ""):
        self.rm = pyvisa.ResourceManager(visa_backend)
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

    def get_power(self) -> float:
        """Read back the current output power level in dBm."""
        return float(self.inst.query("SOUR:POW:POW?"))

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


# -------------------- DMM --------------------
class DMM6500:
    """Keithley DMM6500 wrapper using SCPI over VISA.

    Supports DC/AC voltage and DC/AC current measurements.
    Default function is DC voltage; call ``configure()`` to change.
    """

    FUNCTIONS = {
        "volt:dc": "VOLT:DC",
        "volt:ac": "VOLT:AC",
        "curr:dc": "CURR:DC",
        "curr:ac": "CURR:AC",
        "res": "RES",
    }

    def __init__(self, resource: str, function: str = "volt:dc", visa_backend: str = ""):
        self.rm = pyvisa.ResourceManager(visa_backend)
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = 10_000
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"
        try:
            idn = self.inst.query("*IDN?")
        except Exception:
            idn = "<IDN? failed>"
        print("Connected to:", idn)
        self.configure(function)

    def configure(self, function: str = "volt:dc", auto_range: bool = True) -> None:
        """Select measurement function and optionally enable auto-range."""
        key = function.lower()
        if key not in self.FUNCTIONS:
            raise ValueError(f"Unknown function '{function}'. Choose from: {list(self.FUNCTIONS)}")
        scpi_func = self.FUNCTIONS[key]
        self.inst.write(f':SENS:FUNC "{scpi_func}"')
        if auto_range:
            self.inst.write(f":SENS:{scpi_func}:RANG:AUTO ON")
        self._function = scpi_func

    def measure(self) -> float:
        """Trigger a single reading and return the result."""
        return float(self.inst.query(":READ?"))

    def measure_voltage_dc(self) -> float:
        """Convenience: switch to DC voltage and read once."""
        self.configure("volt:dc")
        return self.measure()

    def measure_voltage_ac(self) -> float:
        """Convenience: switch to AC voltage and read once."""
        self.configure("volt:ac")
        return self.measure()

    def close(self) -> None:
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


# -------------------- Tektronix Oscilloscope --------------------
class TBS1072B:
    """Tektronix TBS1072B wrapper (SCPI over VISA) for Vpp measurements."""

    def __init__(
        self,
        resource: str,
        channel: int = 1,
        timeout_ms: int = 10_000,
        visa_backend: str = "",
    ):
        self.rm = pyvisa.ResourceManager(visa_backend)
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = timeout_ms
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"
        try:
            idn = self.inst.query("*IDN?")
        except Exception:
            idn = "<IDN? failed>"
        print("Connected to:", idn)
        self.set_channel(channel)

    @staticmethod
    def _validate_channel(channel: int) -> int:
        ch = int(channel)
        if ch not in (1, 2):
            raise ValueError("TBS1072B channel must be 1 or 2")
        return ch

    def _query_float(self, cmd: str) -> float:
        raw = self.inst.query(cmd).strip()
        # Some scopes return extras like "9.87E-1 V" or CSV-like tokens.
        token = raw.split(",")[0].split(" ")[0]
        return float(token)

    def set_channel(self, channel: int) -> None:
        """Set the active measurement channel (CH1/CH2)."""
        self.channel = self._validate_channel(channel)
        self.inst.write(f"DATA:SOURCE CH{self.channel}")

    def set_timebase(self, seconds_per_div: float) -> None:
        """Set horizontal scale in seconds/division."""
        if seconds_per_div <= 0:
            raise ValueError("seconds_per_div must be > 0")
        self.inst.write(f"HORizontal:MAIn:SCAle {seconds_per_div}")

    def set_vertical_scale(self, volts_per_div: float, channel: int | None = None) -> None:
        """Set vertical scale in volts/division for CH1/CH2."""
        if volts_per_div <= 0:
            raise ValueError("volts_per_div must be > 0")
        if channel is not None:
            self.set_channel(channel)
        self.inst.write(f"CH{self.channel}:SCAle {volts_per_div}")

    def set_averaging(self, num_averages: int = 64) -> int:
        """Enable/disable acquisition averaging and return applied average count.

        TBS scopes typically support powers of two (2..512). A value <= 1 disables
        averaging (sample mode).
        """
        n = int(num_averages)
        if n <= 1:
            self.inst.write("ACQuire:MODe SAMple")
            return 1

        # Common supported counts on TBS family.
        allowed = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=int)
        n_applied = int(allowed[np.argmin(np.abs(allowed - n))])
        self.inst.write("ACQuire:MODe AVErage")
        self.inst.write(f"ACQuire:NUMAVg {n_applied}")
        return n_applied

    def set_timebase_for_frequency(self, frequency_hz: float, cycles_on_screen: float = 8.0) -> float:
        """Set timebase so about `cycles_on_screen` periods span all 10 divisions.

        Returns the requested seconds/division value.
        """
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")
        if cycles_on_screen <= 0:
            raise ValueError("cycles_on_screen must be > 0")

        seconds_per_div = cycles_on_screen / (10.0 * float(frequency_hz))
        self.set_timebase(seconds_per_div)
        return seconds_per_div

    def measure_vpp(self, channel: int | None = None) -> float:
        """Measure peak-to-peak voltage (Vpp) on the selected channel."""
        if channel is not None:
            self.set_channel(channel)

        # Prefer immediate measurement path.
        # Tek variants differ slightly in accepted source command syntax.
        self.inst.write("MEASUrement:IMMed:TYPE PK2PK")
        try:
            self.inst.write(f"MEASUrement:IMMed:SOUrce CH{self.channel}")
        except Exception:
            self.inst.write(f"MEASUrement:IMMed:SOUrce1 CH{self.channel}")

        for cmd in ("MEASUrement:IMMed:VALue?", "MEASUrement:MEAS1:VALue?"):
            try:
                return self._query_float(cmd)
            except Exception:
                continue
        raise RuntimeError("Failed to read Vpp from TBS1072B with known SCPI queries")

    def close(self) -> None:
        self.inst.close()

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


# -------------------- Power-level scan --------------------
def run_power_scan(
    sg_resource: str,
    dmm_resource: str,
    freq_hz: float,
    p_start: float,
    p_stop: float,
    n_points: int = 30,
    settle_s: float = 0.1,
    dmm_function: str = "volt:dc",
    n_avg: int = 1,
    output_file: str = "power_scan.csv",
) -> None:
    """Sweep SMC100A output level in dBm and record DMM6500 voltage at each step.

    Parameters
    ----------
    sg_resource:   VISA resource string for the SMC100A.
    dmm_resource:  VISA resource string for the Keithley DMM6500.
    freq_hz:       Fixed RF output frequency in Hz during the scan.
    p_start:       Start output power in dBm (inclusive).
    p_stop:        Stop output power in dBm (inclusive).
    n_points:      Number of linearly-spaced power steps.
    settle_s:      Wait time (seconds) after setting each power level.
    dmm_function:  DMM measurement function ('volt:dc', 'volt:ac', etc.).
    n_avg:         Number of DMM readings to average per power step.
    output_file:   CSV path for results.
    """
    powers_dbm = np.linspace(p_start, p_stop, n_points)
    actual_powers: list[float] = []
    voltages: list[float] = []

    with SMC100A(sg_resource) as sg, DMM6500(dmm_resource, function=dmm_function) as dmm:
        sg.set_freq(freq_hz)
        sg.rf_on()
        try:
            for p in powers_dbm:
                sg.set_power(p)
                time.sleep(settle_s)

                actual_p = sg.get_power()
                actual_powers.append(actual_p)

                readings = [dmm.measure() for _ in range(n_avg)]
                v = float(np.mean(readings))
                voltages.append(v)

                print(f"Set: {p:+.2f} dBm  Actual: {actual_p:+.2f} dBm  V: {v:.6g}")
        finally:
            sg.rf_off()

    data = np.column_stack((powers_dbm, actual_powers, voltages))
    np.savetxt(
        output_file,
        data,
        delimiter=",",
        header="SetPower(dBm),ActualPower(dBm),Voltage(V)",
        comments="",
    )
    print(f"Data saved to {output_file}")

    fig, ax1 = plt.subplots()
    ax1.plot(actual_powers, voltages, marker="o", color="tab:blue")
    ax1.set_xlabel("RF output power [dBm]")
    ax1.set_ylabel("Measured voltage [V]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)
    plt.title(f"Power scan @ {freq_hz/1e6:.3f} MHz")
    plt.tight_layout()
    plt.show()


# -------------------- Command-line interface --------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Instrument control: SMC100A + Digilent scope / Keithley DMM6500")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- sub-command: sweep (frequency sweep, SMC100A + Digilent scope) ----
    p_sweep = subparsers.add_parser("sweep", help="Log-spaced frequency sweep with Digilent scope")
    p_sweep.add_argument("--sg", type=str, required=True, help="SMC100A VISA resource string")
    p_sweep.add_argument("--fstart", type=float, default=9e3, help="Start frequency (Hz)")
    p_sweep.add_argument("--fstop", type=float, default=100e6, help="Stop frequency (Hz)")
    p_sweep.add_argument("--npoints", type=int, default=50, help="Number of frequency points")
    p_sweep.add_argument("--channel", type=int, default=1, help="Scope channel to measure")
    p_sweep.add_argument("--outfile", type=str, default="sweep.csv", help="CSV output file")
    p_sweep.add_argument("--ncycles", type=int, default=10, help="Signal cycles to capture per point")

    # ---- sub-command: power-scan (dBm scan, SMC100A + Keithley DMM6500) ----
    p_pscan = subparsers.add_parser("power-scan", help="Sweep SMC100A output level and record DMM6500 voltage")
    p_pscan.add_argument("--sg", type=str, required=True, help="SMC100A VISA resource string")
    p_pscan.add_argument("--dmm", type=str, required=True, help="DMM6500 VISA resource string")
    p_pscan.add_argument("--freq", type=float, default=100e6, help="Fixed RF frequency (Hz)")
    p_pscan.add_argument("--pstart", type=float, default=-20.0, help="Start power (dBm)")
    p_pscan.add_argument("--pstop", type=float, default=10.0, help="Stop power (dBm)")
    p_pscan.add_argument("--npoints", type=int, default=30, help="Number of power steps")
    p_pscan.add_argument("--settle", type=float, default=0.1, help="Settle time per step (s)")
    p_pscan.add_argument("--func", type=str, default="volt:dc",
                         choices=list(DMM6500.FUNCTIONS), help="DMM measurement function")
    p_pscan.add_argument("--navg", type=int, default=1, help="DMM readings to average per step")
    p_pscan.add_argument("--outfile", type=str, default="power_scan.csv", help="CSV output file")

    args = parser.parse_args()

    if args.command == "sweep":
        run_sweep(
            args.sg,
            args.fstart,
            args.fstop,
            n_points=args.npoints,
            channel=args.channel,
            output_file=args.outfile,
            n_cycles=args.ncycles,
        )
    elif args.command == "power-scan":
        run_power_scan(
            sg_resource=args.sg,
            dmm_resource=args.dmm,
            freq_hz=args.freq,
            p_start=args.pstart,
            p_stop=args.pstop,
            n_points=args.npoints,
            settle_s=args.settle,
            dmm_function=args.func,
            n_avg=args.navg,
            output_file=args.outfile,
        )