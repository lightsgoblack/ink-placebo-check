"""tools.tripwire: standalone package for the Amendment E1 text tripwire.

In plain English: a smoke alarm for AI that invents rows of letters on blank papyrus. Point it at a model's
prediction map and a mask of the region that is supposed to be blank; it tells you, in numbers only, whether
that "ink" looks structured like rows of writing rather than random noise or clutter -- never in images.

This package is a thin wrapper (CLI + calibration helper) around the frozen statistical engine in
tools/margin/tripwire_e1.py, which it imports unchanged. See README.md in this directory for the full explanation,
the frozen rule, and the calibration evidence. See tools/margin/tripwire_e1.py's own docstring for the underlying
math (component-shuffle null, row-periodicity statistic T1, letter-window statistic S).
"""
