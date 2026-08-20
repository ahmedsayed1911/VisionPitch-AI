"""Multi-object tracking with camera motion compensation."""

from visionpitch.tracking.gmc import GlobalMotionCompensator
from visionpitch.tracking.postprocess import clean_tracks
from visionpitch.tracking.tracker import MultiObjectTracker, TrackState

__all__ = ["GlobalMotionCompensator", "MultiObjectTracker", "TrackState", "clean_tracks"]
