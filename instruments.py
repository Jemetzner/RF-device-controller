"""
Instrument control utilities for mode-splitting experiments.

This module provides simple wrappers around:
- Rhode & Schwarz SMC100A signal generator
- Tektronix DPO2012 oscilloscope

It uses PyVISA for communication.
"""

from __future__ import annotations

from typing import Optional

import pyvisa


class SignalGeneratorSMC100A:
    """Simple wrapper for R&S SMC100A signal generator via VISA.

    Parameters
    ----------
    resource : str
        VISA resource string, e.g. "TCPIP0::192.168.1.100::inst0::INSTR"
        or "GPIB0::10::INSTR".
    rm : Optional[pyvisa.ResourceManager]
        Existing ResourceManager to reuse; if None, a new one is created.
    """

    def __init__(self, resource: str, rm: Optional[pyvisa.ResourceManager] = None) -> None:
        self._rm = rm or pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource)
        # Reasonable default timeout (ms)
        self._inst.timeout = 5000
        # Enable RF output by default
        self._inst.write("OUTP:STATE ON")

    def set_frequency_and_amplitude(self, frequency_hz: float, amplitude_vrms: float) -> None:
        """Set RF frequency (Hz) and output amplitude (V RMS).

        Adjust units here if your lab prefers dBm or Vpp.
        """
        # Frequency
        self._inst.write(f"FREQ {frequency_hz}")
        # Amplitude in VRMS (change to VPP or DBM if desired)
        self._inst.write("SOUR:POW:UNIT VRMS")
        self._inst.write(f"SOUR:VOLT {amplitude_vrms}")

    def close(self) -> None:
        """Close the VISA session for this generator."""
        try:
            self._inst.close()
        except Exception:
            # Best-effort close; ignore I/O errors on shutdown
            pass


class OscilloscopeDPO2012:
    """Minimal Tektronix DPO2012 wrapper.

    Provides channel peak-to-peak reading and automatic time scale update
    based on the signal generator frequency.
    """

    def __init__(self, resource: str, rm: Optional[pyvisa.ResourceManager] = None) -> None:
        self._rm = rm or pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource)
        self._inst.timeout = 5000
        # Use ASCII for simplicity
        self._inst.write("DATA:ENC ASCIi")

    def set_time_scale_from_frequency(self, frequency_hz: float, periods_on_screen: float = 5.0) -> None:
        """Set horizontal scale such that ~periods_on_screen periods span 10 divisions.

        Tektronix horizontal scale is seconds per division; there are 10 horizontal divisions.

        time_scale = (periods_on_screen / 10) * (1 / frequency_hz)
        """
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")
        period = 1.0 / frequency_hz
        time_scale = (periods_on_screen / 10.0) * period
        self._inst.write(f"HOR:MAIN:SCA {time_scale}")

    def measure_vpp(self, channel: int = 1) -> float:
        """Return peak-to-peak voltage for a given channel.

        Uses the oscilloscope's built-in measurement system.
        """
        if channel not in (1, 2):
            raise ValueError("DPO2012 has channels 1 and 2")

        # Configure immediate measurement for peak-to-peak on the selected channel
        self._inst.write(f"MEASU:IMM:SOU CH{channel}")
        self._inst.write("MEASU:IMM:TYPE PK2PK")
        vpp_str = self._inst.query("MEASU:IMM:VAL?")

        try:
            return float(vpp_str)
        except ValueError as exc:
            raise RuntimeError(f"Could not parse Vpp reading from scope: {vpp_str!r}") from exc

    def close(self) -> None:
        """Close the VISA session for this scope."""
        try:
            self._inst.close()
        except Exception:
            pass


class ModeSplittingInstruments:
    """Convenience wrapper tying the generator to the scope.

    On each frequency change, the scope time scale is adjusted automatically.
    """

    def __init__(
        self,
        sig_gen_resource: str,
        scope_resource: str,
        rm: Optional[pyvisa.ResourceManager] = None,
    ) -> None:
        self._rm = rm or pyvisa.ResourceManager()
        self.sig_gen = SignalGeneratorSMC100A(sig_gen_resource, rm=self._rm)
        self.scope = OscilloscopeDPO2012(scope_resource, rm=self._rm)

    def set_frequency_and_amplitude(
        self,
        frequency_hz: float,
        amplitude_vrms: float,
        *,
        periods_on_screen: float = 5.0,
    ) -> None:
        """Set generator frequency and amplitude, then retune scope time scale."""
        self.sig_gen.set_frequency_and_amplitude(frequency_hz, amplitude_vrms)
        self.scope.set_time_scale_from_frequency(frequency_hz, periods_on_screen=periods_on_screen)

    def measure_vpp(self, channel: int = 1) -> float:
        """Return peak-to-peak voltage on the chosen scope channel."""
        return self.scope.measure_vpp(channel=channel)

    def close(self) -> None:
        """Close both instruments."""
        self.sig_gen.close()
        self.scope.close()


__all__ = [
    "SignalGeneratorSMC100A",
    "OscilloscopeDPO2012",
    "ModeSplittingInstruments",
]


