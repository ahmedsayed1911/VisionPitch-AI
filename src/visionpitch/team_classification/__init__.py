"""Automatic team, role and goalkeeper discovery."""

from visionpitch.team_classification.classifier import TeamClassifier, TeamDiscoveryReport
from visionpitch.team_classification.crops import JerseyCropExtractor

__all__ = ["JerseyCropExtractor", "TeamClassifier", "TeamDiscoveryReport"]
