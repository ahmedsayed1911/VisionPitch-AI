"""Human-reviewed broadcast annotation workflow.

Model predictions live in this package only as *proposals* that reduce clicking.
Ground truth comes from a person, is stored in separate files, and never
overwrites a prediction or is overwritten by one.
"""

from visionpitch.annotation.broadcast_audit import AuditResult, Shot, ShotType

__all__ = ["AuditResult", "Shot", "ShotType"]
