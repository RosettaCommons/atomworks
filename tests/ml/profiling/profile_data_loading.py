"""
Simple script to profile the data loading pipeline. Designed to be run from the command line.

+-----------------------------------------------------------------------------------------+
To profile PERFORMANCE with cProfile:
    python -m cProfile -o profile_data_loading_pipeline.prof profile_data_loading.py
+-----------------------------------------------------------------------------------------+
To profile MEMORY with memray:
    memray run profile_data_loading.py
Then run the printed command to generate the flame fraph from the Memray output file.
+-----------------------------------------------------------------------------------------+
"""

import logging

from tqdm import tqdm

from atomworks.ml.datasets import get_row_and_index_by_example_id
from tests.conftest import PDB_DATASET
from tests.pipelines.test_data_loading_pipelines import BENCHMARK_EXAMPLE_IDS


def load_examples(indices):
    for idx in tqdm(indices):
        _ = PDB_DATASET[idx]


if __name__ == "__main__":
    # Turn off logging
    logging.disable(logging.CRITICAL)

    # Get the indices of the pre-selected examples
    indices = []
    for example_id in BENCHMARK_EXAMPLE_IDS[:2]:
        index = get_row_and_index_by_example_id(PDB_DATASET, example_id)["index"]
        indices.append(index)

    # Load the examples
    load_examples(indices)
