"""Condition system for StandardAnnotations that are intended to be passed to ML models."""

from atomworks.io.utils.standard_annotations.base import NO_DEFAULT
from atomworks.ml.conditions import definitions
from atomworks.ml.conditions.base import (
    CONDITIONS,
    ConditionBase,
)

# --- alias table ---
Condition = CONDITIONS
C_SEQ = CONDITIONS.sequence
C_IDX = CONDITIONS.index
C_DIS = CONDITIONS.distance
C_CRD = CONDITIONS.coordinate
C_CHA = CONDITIONS.chain
C_CTR = CONDITIONS.c_terminus
C_NTR = CONDITIONS.n_terminus
# --------------------

# Re-importing for backwards compatibility
# isort: split  # these imports must remain after the alias table above
import sys  # noqa: E402

import atomworks.ml.conditions.definitions as conditions  # noqa: E402
from atomworks.io.utils.standard_annotations.base import Level  # noqa: E402
from atomworks.ml.utils import annotator  # noqa: E402

# Ensure that re-imported modules are registered properly
sys.modules[__name__ + ".annotator"] = annotator
sys.modules[__name__ + ".conditions"] = conditions
