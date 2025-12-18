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
    def __init__(self, channels=[1]):
        self.channels = [ch - 1 for ch in channels]  # 0-indexed
        self.device = dwf.Device()
        self.device.open()
        self.ai = self.device.analog_input
        self.sample_rate = None
        self.buffer_size = None

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

    def set_timebase(self, sample_rate, buffer_size):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.ai.record_length = buffer_size
        self.ai.sampling_rate = sample_rate

    def _wait_for_completion(self, timeout=2.0):
        t0 = time.time()
        while True:
            status = self.ai.read_status(read_data=True)
            if status in (0, 2):  # Done or Prefill
                break
            if time.time() - t0 > timeout:
                raise TimeoutError("Acquisition did not complete in time")
            time.sleep(0.001)

    def get_waveform(self, channel=1, timeout=2.0):
        idx = channel - 1
        if self.sample_rate is None or self.buffer_size is None:
            raise ValueError("Sample rate and buffer size must be set before acquisition")
        self.ai.trigger_source = "none"
        self.ai.single(configure=True, start=True)
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
            period = 1.0 / f
            n_cycles = 10
            display_time = period * n_cycles
            buffer_size = 5000
            sample_rate = buffer_size / display_time
            scope.set_timebase(sample_rate=sample_rate, buffer_size=buffer_size)

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
    device.set_freq("1 MHz")
    device.set_power("0dBm")
    device.rf_on()
    scope = DigilentScope()
    scope.configure_channel(v_range = 0.5)
    scope.set_timebase(sample_rate = 1e-3, buffer_size = 1000)
    trace = scope.get_waveform(channel = 0, timeout=10)
    print(scope.get_peak_to_peak())
    plt.plot(trace)
    plt.show()