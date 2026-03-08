# Physical and grid parameters for ASAP7 7.5-track multi-VT technology.
# Used across timing analysis, buffering, and placement density modules.

RF_RISE = 0   # rise edge index
RF_FALL = 1   # fall edge index

# Moment tensor column indices (M0 = total cap, M1 = first RC moment, M2 = second)
MOM0, MOM1, MOM2 = 0, 1, 2

# Pi-model component indices (C1 near driver, C2 far, Rpi internal)
PI_C1, PI_C2, PI_RPI = 0, 1, 2

# ASAP7 interconnect parameters (SI units, per meter)
WIRE_R = 3.23151e07    # sheet resistance [Ohm/m]
WIRE_C = 1.73323e-10   # line capacitance  [F/m]

# Database unit to meter conversion
DBU_NM = 1e-9

# Number of bins along each axis for placement density grid
DENSITY_BINS = 64
