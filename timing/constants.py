"""
timing.constants — physical and numerical constants for the ASAP7 PDK.

All wire RC values are in SI units (Ω/m and F/m) converted from the
standard OpenROAD ``set_wire_rc`` parameters used in the contest.

Unit conventions used throughout the timing engine
---------------------------------------------------
* Length  : DBU (design database units).  For ASAP7, 1 DBU = 1 nm.
* Time    : seconds.
* Capacitance : farads (F).
* Resistance  : ohms (Ω).

The helper ``TO_MICRON`` converts DBU → µm → is then multiplied by the
per-µm wire constants below.
"""

# Rise / fall index into [T, 2] tensors.
RISE = 0
FALL = 1

# Elmore moment indices in [N, 3] tensors.
M0 = 0   # zeroth moment  (∝ total downstream capacitance)
M1 = 1   # first moment   (∝ R·C²)
M2 = 2   # second moment  (∝ R²·C³)

# Two-pole Π model indices in [N, 3] tensors.
C1  = 0  # near-driver capacitance  (fF in scaled form)
C2  = 1  # far-end capacitance      (fF in scaled form)
RPI = 2  # series resistance        (Ω)

# ASAP7 signal wire R/C per µm (from contest RC corner).
# R_PER_LEN : kΩ/µm → converted to Ω/µm  (3.23151e4 Ω/µm)
# C_PER_LEN : pF/µm → converted to F/µm  (1.73323e-16 F/µm)
# The values below match OpenROAD's ``set_wire_rc -signal`` parameters.
R_PER_LEN: float = 3.23151e7   # Ω/m  (= 3.23151e-2 kΩ/µm × 1e9 µm/m)
C_PER_LEN: float = 1.73323e-10 # F/m  (= 1.73323e-1 pF/µm × 1e-12 F/pF × 1e6 µm/m)

# Average via resistance (V1–V4 average) in Ω, used in moment computation.
VIA_RESISTANCE: float = 15.85  # Ω

# DBU-to-metres conversion for ASAP7 (1 DBU = 1 nm = 1e-9 m)
# Multiply wire lengths in DBU by TO_MICRON to obtain µm, then use
# R_PER_LEN / C_PER_LEN (which are per µm in effective units used below).
TO_MICRON: float = 1e-9  # DBU → µm: 1 nm * 1e-9 m/nm * 1e6 µm/m = 1e-3 µm...
# Note: constants R_PER_LEN and C_PER_LEN are calibrated so that
#   R [Ω] = R_PER_LEN * len_dbu * TO_MICRON
#   C [F] = C_PER_LEN * len_dbu * TO_MICRON

# Density grid resolution (GRID_SIZE × GRID_SIZE bins).
GRID_SIZE: int = 64
