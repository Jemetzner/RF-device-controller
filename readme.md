## RF-devices – Instrument Control and Sweeps

This repository provides a small, reproducible setup for:

- **Rohde & Schwarz SMC100A** RF signal generator (via VISA)
- **Digilent** scope (via `dwfpy`)

with helpers for **automatic timebase selection** and **frequency sweeps**.

The project is managed with **Poetry**.

---

## 1. Installation with Poetry

### 1.1. Clone and create the environment

From the folder that contains this project:

```bash
cd path_to_clone
poetry install
```

This will create a virtual environment and install:

- `pyvisa`, `pyusb`, `rsinstrument` – VISA / instrument support
- `numpy`, `matplotlib`, `dwfpy` – analysis and Digilent scope support
- `ipython`, `jupyterlab`, `notebook` – for interactive work (dev group)

### 1.2. Activating the environment

You can either run commands through Poetry:

```bash
poetry run python
poetry run jupyter lab
```

or spawn a shell inside the env:

```bash
poetry shell
python
```

---

## 2. VISA backends and Rohde & Schwarz RS-VISA

The code in `instruments.py` talks to the SMC100A using **SCPI over VISA**. For that to work, you need **a VISA implementation** installed on your system:

- **R&S RS‑VISA** (recommended for R&S gear), or
- **NI‑VISA** (National Instruments), or
- another VISA implementation compatible with `pyvisa`.

Once RS‑VISA (or NI‑VISA) is installed:

1. Use the **R&S VISA Assistant** (or NI Measurement & Automation Explorer) to verify that the SMC100A is visible and note its **resource string**, e.g.  
   `TCPIP0::192.168.1.100::INSTR` or `USB::0x0AAD::0x006E::102502::INSTR`.
2. You can then use that string directly in both **`pyvisa`** and **`rsinstrument`**.

### 2.1. Using `rsinstrument` (high-level R&S API)

Besides the `pyvisa`‑based `SMC100A` class in `instruments.py`, you can also talk to the generator using the official R&S Python driver `rsinstrument`, which is already listed as a dependency.

Example (standalone, not using `instruments.py`):

```python
from rsinstrument import RsInstrument

resource = "TCPIP0::192.168.1.100::INSTR"  # replace with your SMC100A address

inst = RsInstrument(
    resource,
    id_query=True,
    reset=False,
    options="SelectVisa=rs"  # force RS-VISA if multiple VISA backends are installed
)

print("Connected to:", inst.idn_string)

# Set frequency and level
inst.write_str("SOUR:FREQ 10e6")       # 10 MHz
inst.write_str("SOUR:POW:POW 0 dBm")   # 0 dBm
inst.write_str("OUTP ON")              # RF on

# Read back to verify
print("Freq:", inst.query_str("SOUR:FREQ?"))
print("Power:", inst.query_str("SOUR:POW:POW?"))

inst.write_str("OUTP OFF")
inst.close()
```

If you prefer, you can later re‑implement the `SMC100A` class on top of `RsInstrument` instead of `pyvisa`, but the existing code will already work with RS‑VISA because `pyvisa` simply calls into the installed VISA library.

---

## 3. Using the provided instrument classes

All the convenience logic lives in `instruments.py`.

### 3.1. Simple RF generator usage (SMC100A)

```python
from instruments import SMC100A

sg = SMC100A("USB::0x0AAD::0x006E::102502::INSTR")  # replace with your resource

sg.set_freq(10e6)       # 10 MHz
sg.set_power(0)         # 0 dBm
sg.rf_on()

# do experiment...

sg.rf_off()
sg.close()
```

You can also pass SCPI strings such as `"10000 kHz"` to `set_freq`, and `"0 dBm"` to `set_power` if you prefer explicit units.

### 3.2. Digilent scope usage (timebase and Vpp)

```python
from instruments import DigilentScope, choose_timebase

frequency_hz = 10e6
fs, buf = choose_timebase(frequency_hz, n_cycles=10)

scope = DigilentScope(sample_rate=fs, buffer_size=buf)
scope.configure_channel(channel=1, v_range=0.1, offset=0.0)

trace = scope.get_waveform(channel=1, timeout=2.0)
vpp = scope.get_peak_to_peak(channel=1)

print("Vpp:", vpp)
scope.close()
```

The helpers:

- `choose_timebase(f_signal, n_cycles)` → `(sample_rate, buffer_size)`  
  picks a sample rate and power‑of‑two buffer so that you see roughly `n_cycles` of the signal with ~5000 samples.

### 3.3. Combined sweep (SMC100A + Digilent scope)

The high‑level sweep function is `run_sweep`:

```python
from instruments import run_sweep

run_sweep(
    sg_resource="USB::0x0AAD::0x006E::102502::INSTR",  # SMC100A VISA resource
    f_start=10e3,                                     # 10 kHz
    f_stop=10e6,                                      # 10 MHz
    n_points=50,
    channel=1,
    output_file="sweep.csv",
    n_cycles=10,                                      # cycles captured per point
)
```

This will:

- Log‑space sweep from `f_start` → `f_stop`
- For each point:
  - Set the SMC100A frequency
  - Adjust Digilent scope `sample_rate` + `buffer_size` to capture ~`n_cycles`
  - Measure Vpp on `channel`
- Save CSV (`Frequency(Hz),Vpp(V)`) and show a semi‑log plot.

### 3.4. Command‑line sweep

You can also run the sweep as a script:

```bash
poetry run python instruments.py \
  --sg "USB::0x0AAD::0x006E::102502::INSTR" \
  --fstart 1e4 \
  --fstop 1e7 \
  --npoints 50 \
  --channel 1 \
  --outfile sweep.csv \
  --ncycles 10
```

---

## 4. Using in notebooks

Inside a Jupyter notebook started with `poetry run jupyter lab`:

```python
from instruments import SMC100A, DigilentScope, choose_timebase

sg = SMC100A("USB::0x0AAD::0x006E::102502::INSTR")
fs, buf = choose_timebase(10e6, n_cycles=10)
scope = DigilentScope(sample_rate=fs, buffer_size=buf)
scope.configure_channel(channel=1, v_range=0.1)

trace = scope.get_waveform(channel=1)
vpp = scope.get_peak_to_peak(channel=1)

print("Vpp:", vpp)
```

This integrates naturally with your existing analysis workflows (e.g. exporting to CSV and feeding into Lorentzian fitting code).

---

If you’d like, the next step can be to add an example notebook that performs a sweep and immediately runs your Lorentzian fit pipeline on the acquired data.


