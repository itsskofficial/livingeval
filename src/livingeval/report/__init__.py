"""Versioned records, markdown tables and figures drawn from them."""

from livingeval.report.record import Record, load, save
from livingeval.report.tables import (
    coverage_table,
    decay_table,
    false_alarm_table,
    ladder_table,
)


def figures(record, out_dir):
    """Draw every figure the record supports. Needs `livingeval[figures]`."""
    from livingeval.report.figures import all_figures

    return all_figures(record, out_dir)


__all__ = [
    "Record",
    "coverage_table",
    "decay_table",
    "false_alarm_table",
    "figures",
    "ladder_table",
    "load",
    "save",
]
