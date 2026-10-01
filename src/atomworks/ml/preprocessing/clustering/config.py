"""Configuration for MMseqs2 sequence clustering."""

from dataclasses import dataclass
from enum import IntEnum


class CoverageMode(IntEnum):
    """MMseqs2 coverage modes."""

    BIDIRECTIONAL = 0
    TARGET = 1
    QUERY = 2


class ClusterMode(IntEnum):
    """MMseqs2 cluster modes."""

    GREEDY_SET_COVER = 0
    CONNECTED_COMPONENT = 1
    GREEDY_INCREMENTAL = 2


@dataclass(frozen=True)
class MMseqs2Config:
    """Configuration for MMseqs2 easy-cluster."""

    cluster_identity: float = 0.4
    coverage: float = 0.8
    sensitivity: float = 8.0
    cluster_mode: ClusterMode = ClusterMode.GREEDY_SET_COVER
    coverage_mode: CoverageMode = CoverageMode.BIDIRECTIONAL

    def __post_init__(self) -> None:
        if not 0.0 <= self.cluster_identity <= 1.0:
            raise ValueError(f"cluster_identity must be in [0, 1], got {self.cluster_identity}")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")
