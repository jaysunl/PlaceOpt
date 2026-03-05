"""eda package — EDA utilities: buffer insertion and placement legalization."""

from placeopt.eda.buffering import (
    is_buffer, is_inverter,
    insert_buffer, commit_buffering, remove_buffer,
    BufferInsertionResult,
)
from placeopt.eda.placement import run_detailed_placement

__all__ = [
    "is_buffer", "is_inverter",
    "insert_buffer", "commit_buffering", "remove_buffer",
    "BufferInsertionResult",
    "run_detailed_placement",
]
