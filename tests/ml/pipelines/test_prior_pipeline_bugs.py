import logging

import pytest

from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from tests.ml.conftest import (
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
    TEMPLATE_DIR,
    TEMPLATE_LOOKUP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################################
# ------------------- AF3 Pipeline -----------------------#
##########################################################

PRIOR_PIPELINE_BUGS_AF3 = ["7qbs", "5epq", "2g37", "4v4s"]


@pytest.mark.parametrize("pdb_id", PRIOR_PIPELINE_BUGS_AF3)
@pytest.mark.slow
def test_prior_pipeline_bugs_af3(pdb_id: str):
    """Run a single example through the pipeline. Useful for debugging specific examples."""

    input = cached_parse(pdb_id)
    input["example_id"] = pdb_id

    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        pipe = build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        )
        output = pipe(input)

    assert output is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-m not very_slow"])
