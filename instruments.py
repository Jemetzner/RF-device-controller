#!/usr/bin/env python3
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pyvisa
import dwfpy as dwf

# -------------------- Signal Generator --------------------
class SMC100A:
    def __init__(self, resource):
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(resource)
        self.inst.timeout = 10000
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"
        print("Connected to:", self.inst.query("*IDN?"))

    def set_freq(self, hz):
        self.inst.write(f"SOUR:FREQ {hz}")

    def set_power(self, dbm):
        self.inst.write(f"SOUR:POW:POW {dbm}")

    def rf_on(self):
        self.inst.write("OUTP ON")

    def rf_off(self):
        self.inst.write("OUTP OFF")

    def close(self):
        try:
            self.rf_off()
        finally:
            self.inst.close()

# -------------------- Oscilloscope --------------------
class DigilentScope:
    def __init__(self, channels=[1], sample_rate = 20e6, buffer_size = 8192):
        self.channels = [ch - 1 for ch in channels]  # 0-indexed
        self.device = dwf.Device()
        self.device.open()
        self.ai = self.device.analog_input
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size


        for idx in self.channels:
            ch = self.ai[idx]
            ch.enabled = True
            ch.range = 5.0
            ch.offset = 0.0

    def configure_channel(self, channel=1, v_range=5.0, offset=0.0):
        idx = channel - 1
        ch = self.ai[idx]
        ch.enabled = True
        ch.range = v_range
        ch.offset = offset



    def _wait_for_completion(self, timeout=2.0):
        t0 = time.time()
        while True:
            status = self.ai.read_status(read_data=True)
            if status in (0, 2):  # Done or Prefill
                break
            if time.time() - t0 > timeout:
                raise TimeoutError("Acquisition did not complete in time")
            time.sleep(0.001)

    def get_waveform(self,channel = 1, timeout=2.0):
        idx = channel - 1
        self.ai.setup_edge_trigger(mode="auto", channel=channel, slope="rising", level=0.1, hysteresis=0.01)
        self.ai.single(configure=True, start=True, buffer_size = self.buffer_size, sample_rate = self.sample_rate)
        self._wait_for_completion(timeout=timeout)
        samples = np.array(self.ai[idx].get_data())
        return samples

    def get_peak_to_peak(self, channel=1, timeout=2.0):
        data = self.get_waveform(channel, timeout=timeout)
        return np.max(data) - np.min(data)

    def close(self):
        self.device.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



VALID_MANTISSAS = np.array([1, 2, 5])

def quantize_sample_rate(fs_ideal, fs_min=0.1, fs_max=100e6):
    """
    Snap fs_ideal to nearest valid Digilent sample rate
    (1,2,5 × 10^n)
    """
    fs_ideal = np.clip(fs_ideal, fs_min, fs_max)

    decade = np.floor(np.log10(fs_ideal))
    candidates = []

    for d in [decade - 1, decade, decade + 1]:
        for m in VALID_MANTISSAS:
            candidates.append(m * 10**d)

    candidates = np.array(candidates)
    candidates = candidates[(candidates >= fs_min) & (candidates <= fs_max)]

    return candidates[np.argmin(np.abs(candidates - fs_ideal))]
VALID_BUFFERS = np.array([2**n for n in range(5, 15)])  # 32 → 16384

def select_buffer_size(fs, f_signal, n_cycles=10):
    """
    Pick the smallest power-of-two buffer that captures
    at least n_cycles of the signal
    """
    required = fs * (n_cycles / f_signal)

    for buf in VALID_BUFFERS:
        if buf >= required:
            return buf

    return VALID_BUFFERS[-1]  # clamp to max

def choose_timebase(f_signal, n_cycles=10):
    """
    Returns (sample_rate, buffer_size)
    respecting Digilent constraints
    """
    # Ideal continuous math
    display_time = n_cycles / f_signal
    fs_ideal = 5000 / display_time

    fs = quantize_sample_rate(fs_ideal)
    buf = select_buffer_size(fs, f_signal, n_cycles)

    return fs, buf



# -------------------- Sweep Script --------------------
def run_sweep(sg_resource, f_start, f_stop, n_points=50, channel=1, output_file="sweep.csv"):
    # Connect instruments
    sg = SMC100A(sg_resource)
    scope = DigilentScope(channels=[channel])
    scope.configure_channel(channel=channel, v_range=5.0, offset=0.0)

    frequencies = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
    vpp_results = []

    sg.rf_on()
    try:
        for f in frequencies:
            sg.set_freq(f)


            # Timebase: capture ~10 periods on 5000 points
            fs, buf = choose_timebase(f, n_cycles=10)
            scope.set_timebase(sample_rate=fs, buffer_size=buf)

            # Small delay to let instruments settle
            time.sleep(0.01)

            # Acquire Vpp
            try:
                vpp = scope.get_peak_to_peak(channel=channel, timeout=2.0)
            except Exception as e:
                print(f"Error at {f/1e6:.3f} MHz: {e}")
                vpp = np.nan
            vpp_results.append(vpp)
            print(f"Freq: {f/1e6:.3f} MHz -> Vpp: {vpp:.3f} V")

    finally:
        sg.rf_off()
        sg.close()
        scope.close()

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

    # parser = argparse.ArgumentParser(description="Frequency sweep using SMC100A + Digilent scope")
    # parser.add_argument("--sg", type=str, required=True, help="Signal generator VISA resource string")
    # parser.add_argument("--fstart", type=float, default=9e3, help="Start frequency (Hz)")
    # parser.add_argument("--fstop", type=float, default=100e6, help="Stop frequency (Hz)")
    # parser.add_argument("--npoints", type=int, default=50, help="Number of points in sweep")
    # parser.add_argument("--channel", type=int, default=1, help="Scope channel to measure")
    # parser.add_argument("--outfile", type=str, default="sweep.csv", help="CSV file to save results")
    # args = parser.parse_args()

    # run_sweep(args.sg, args.fstart, args.fstop, n_points=args.npoints, channel=args.channel, output_file=args.outfile)
    device = SMC100A("USB::0x0AAD::0x006E::102502::INSTR")
    f = "10000 kHz"
    device.set_freq(f)
    device.set_power("0dBm")
    device.rf_on()
    fs,buf = choose_timebase(10000e3)
    scope = DigilentScope(sample_rate=fs, buffer_size = buf)
    scope.configure_channel(v_range = 0.1)
    trace = scope.get_waveform(channel = 0, timeout=10)
    print(scope.get_peak_to_peak())
    plt.plot(trace)
    plt.show()