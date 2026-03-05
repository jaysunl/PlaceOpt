"""
placeopt — Post-Placement Gate Sizing & Buffer Insertion
=========================================================

This package implements a gradient-based simultaneous gate-sizing and
buffer-insertion optimizer for post-placement netlists targeting the
ISPD 2026 contest metric (TNS + dynamic power minimization).

Key design decisions
--------------------
* All timing and placement quantities are kept as differentiable PyTorch
  tensors so that a single ``loss.backward()`` propagates gradients through
  the full STA graph, gate-size distributions, and placement positions.
* Three independent Adam optimizers handle gate sizing (``U_flat``),
  buffer insertion (``buffering_tensor``), and cell movement (``cell_xy``).
* Wire delay is modelled via Elmore moments up to second order, which yields
  a two-pole Π load model seen by each gate driver.

Package layout
--------------
circuit/    — circuit data model (gates, pins, nets, Steiner trees)
timing/     — differentiable STA engine (LUTs, moments, Π model, forward pass)
opt/        — gradient optimizer and placement density penalty
eda/        — EDA utilities: buffer insertion and legalization
io/         — design I/O (OpenROAD load/write helpers)
"""

__version__ = "1.0.0"
