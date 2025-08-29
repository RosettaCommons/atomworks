"""ML-specific test fixtures and utilities for atomworks.ml tests."""

import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

from atomworks.constants import AF3_EXCLUDED_LIGANDS_REGEX, _load_env_var
from atomworks.io.tools.inference import SequenceComponent
from atomworks.ml.datasets.datasets import ConcatDatasetWithID, PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers import (
    GenericDFParser,
    InterfacesDFParser,
    PNUnitsDFParser,
    ValidationDFParserLikeAF3,
)
from atomworks.ml.datasets.parsers.base import DEFAULT_CIF_PARSER_ARGS
from atomworks.ml.pipelines.af3 import build_af3_transform_pipeline
from atomworks.ml.pipelines.rf2aa import build_rf2aa_transform_pipeline
from atomworks.ml.preprocessing.constants import TRAINING_SUPPORTED_CHAIN_TYPES_INTS
from atomworks.ml.utils.io import read_parquet_with_metadata
from atomworks.ml.utils.testing import cached_parse
from tests.conftest import TEST_DATA_DIR

##########################################################################################
# + ----------------------------------- Environment ------------------------------------ +
##########################################################################################


def pytest_configure(config):
    # Get the directory where conftest.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to .env file in the parent directory
    dotenv_path = os.path.join(current_dir, "../..", ".env")

    # Load the environment variables
    load_dotenv(dotenv_path)


if not os.environ.get("PDB_MIRROR_PATH") or not os.path.exists(os.environ.get("PDB_MIRROR_PATH")):
    raise pytest.UsageError(
        "ERROR: Required PDB_MIRROR_PATH environment variable not set. "
        "Please set this in the .env file or in your shell environment."
    )


##########################################################################################
# + -------------------------------- Test Decorators ----------------------------------- +
##########################################################################################


@pytest.fixture
def cache_dir():
    """Fixture providing the cache directory path or skipping if not available."""
    cache_dir = _load_env_var("RESIDUE_CACHE_DIR")
    if not cache_dir:
        pytest.skip("RESIDUE_CACHE_DIR environment variable not set")

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        pytest.skip(f"RESIDUE_CACHE_DIR directory not found: {cache_path}")

    return cache_path


##########################################################################################
# + ------------------------------------ Constants ------------------------------------- +
##########################################################################################

TEST_DATA_ML = TEST_DATA_DIR / "ml"

PROTEIN_MSA_DIRS = [
    {
        "dir": str(TEST_DATA_DIR / "shared" / "msa"),
        "extension": ".a3m.gz",
        "directory_depth": 2,
    }
]

RNA_MSA_DIRS = [
    {
        "dir": str(TEST_DATA_DIR / "shared" / "msa"),
        "extension": ".afa",
        "directory_depth": 0,
    }
]

TEMPLATE_DIR = TEST_DATA_DIR / "shared" / "template"
TEMPLATE_LOOKUP = TEST_DATA_DIR / "shared" / "template_lookup.csv"

##########################################################################################
# + ----------------------------------- Dataframes ------------------------------------- +
##########################################################################################


# Interfaces/PN Units
@pytest.fixture(scope="session")
def pn_units_df():
    path = TEST_DATA_ML / "pdb_pn_units" / "metadata.parquet"
    df = read_parquet_with_metadata(path)
    # df.attrs["base_path"] = str(TEST_DATA_ML / "pdb_pn_units" / "cif")
    return df


@pytest.fixture(scope="session")
def interfaces_df():
    path = TEST_DATA_ML / "pdb_interfaces" / "metadata.parquet"
    df = read_parquet_with_metadata(path)
    # df.attrs["base_path"] = str(TEST_DATA_ML / "pdb_interfaces" / "cif")
    return df


# AF2 Distillation Facebook, with and without table-wide metadata (to test metadata handling)
@pytest.fixture(scope="session")
def af2_distillation_facebook_df_no_metadata():
    path = TEST_DATA_ML / "af2_distillation" / "metadata.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def af2_distillation_facebook_df_with_metadata():
    df = read_parquet_with_metadata(TEST_DATA_ML / "af2_distillation" / "metadata.parquet")
    df.attrs["base_path"] = str(TEST_DATA_ML / "af2_distillation" / "cif")
    return df


# Validation
@pytest.fixture(scope="session")
def af3_validation_df():
    return read_parquet_with_metadata(TEST_DATA_ML / "af3_splits_test_metadata.parquet")


##########################################################################################
# + ------------------------------------ Datasets -------------------------------------- +
##########################################################################################

SHARED_TEST_FILTERS = [
    "deposition_date < '2022-01-01'",
    "resolution < 5.0 and ~method.str.contains('NMR')",
    "num_polymer_pn_units <= 20",  # To ensure we don't freeze loading a massive example
    "cluster.notnull()",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"pn_unit_2_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

TEST_DIFFUSION_BATCH_SIZE = 32  # Set to a value other than default (48) for testing

# +--------------------------------------------------------------------------+
# Base PandasDataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def pn_units_pandas_dataset(pn_units_df):
    return PandasDataset(
        name="pn_units",
        id_column="example_id",
        data=pn_units_df,
        filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def interfaces_pandas_dataset(interfaces_df):
    return PandasDataset(
        name="interfaces",
        id_column="example_id",
        data=interfaces_df,
        filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def validation_pandas_dataset(af3_validation_df):
    return PandasDataset(
        name="validation",
        data=af3_validation_df,
        id_column="example_id",
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def distillation_pandas_dataset_no_metadata(af2_distillation_facebook_df_no_metadata):
    return PandasDataset(
        data=af2_distillation_facebook_df_no_metadata,
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash", "path"],
    )


@pytest.fixture(scope="session")
def distillation_pandas_dataset_with_metadata(af2_distillation_facebook_df_with_metadata):
    return PandasDataset(
        data=af2_distillation_facebook_df_with_metadata,
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash", "path"],
    )


# +--------------------------------------------------------------------------+
# RF2AA Dataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def rf2aa_pn_units_dataset(pn_units_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=PNUnitsDFParser(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3,
            crop_spatial_probability=2 / 3,
            convert_feats_to_rf2aa_input_tuple=False,
            assert_rf2aa_assumptions=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=pn_units_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def rf2aa_interfaces_dataset(interfaces_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=InterfacesDFParser(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=1.0,
            crop_contiguous_probability=0.0,
            assert_rf2aa_assumptions=False,
            convert_feats_to_rf2aa_input_tuple=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=interfaces_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def rf2aa_pdb_dataset(rf2aa_pn_units_dataset, rf2aa_interfaces_dataset):
    return ConcatDatasetWithID(datasets=[rf2aa_pn_units_dataset, rf2aa_interfaces_dataset])  # NOTE: Order matters!


@pytest.fixture(scope="session")
def rf2aa_validation_dataset(validation_pandas_dataset):
    """Create a StructuralDatasetWrapper for RF2AA validation."""
    return StructuralDatasetWrapper(
        dataset_parser=ValidationDFParserLikeAF3(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
            crop_contiguous_probability=0.0,  # NOTE: Zero probability for cropping ; we don't crop during validation
            assert_rf2aa_assumptions=False,
            convert_feats_to_rf2aa_input_tuple=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=validation_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


# +--------------------------------------------------------------------------+
# AF3 Dataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def af3_pn_units_dataset(pn_units_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=PNUnitsDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3,
            crop_spatial_probability=2 / 3,
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=pn_units_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_interfaces_dataset(interfaces_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=InterfacesDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=1.0,
            crop_contiguous_probability=0.0,
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=interfaces_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_pdb_dataset(af3_pn_units_dataset, af3_interfaces_dataset):
    return ConcatDatasetWithID(datasets=[af3_pn_units_dataset, af3_interfaces_dataset])  # NOTE: Order matters!


@pytest.fixture(scope="session")
def af3_validation_dataset(validation_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=ValidationDFParserLikeAF3(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=True,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
            crop_contiguous_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        dataset=validation_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_dataset_no_metadata(distillation_pandas_dataset_no_metadata):
    return StructuralDatasetWrapper(
        dataset=distillation_pandas_dataset_no_metadata,
        dataset_parser=GenericDFParser(
            base_path=str(TEST_DATA_ML / "af2_distillation" / "cif"),
            extension=".cif",
        ),
        cif_parser_args={},
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=[],
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            is_inference=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_dataset_with_metadata(distillation_pandas_dataset_with_metadata):
    return StructuralDatasetWrapper(
        dataset=distillation_pandas_dataset_with_metadata,
        dataset_parser=GenericDFParser(),
        cif_parser_args={},
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=[],
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            is_inference=False,
            template_lookup_path=TEMPLATE_LOOKUP,
            template_base_dir=TEMPLATE_DIR,
        ),
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_concat_dataset(af3_af2fb_distillation_dataset_no_metadata):
    return ConcatDatasetWithID(datasets=[af3_af2fb_distillation_dataset_no_metadata])


##########################################################################################
# + ------------------------------------ Database Fixtures --------------------------------- +
##########################################################################################


@pytest.fixture
def atom_array():
    """
    Load a CIF file from somewhere local and return the atom_array
    """
    merged_cif_parser_args = {
        **DEFAULT_CIF_PARSER_ARGS,
        **{
            "fix_arginines": False,
            "add_missing_atoms": False,  # this is crucial otherwise the annotations are deleted
        },
    }
    merged_cif_parser_args.pop("add_bond_types_from_struct_conn")
    merged_cif_parser_args.pop("remove_ccds")
    data = cached_parse("6lyz", **merged_cif_parser_args)
    atom_array = data["atom_array"]
    return atom_array


@pytest.fixture
def chemical_components():
    """
    Makes a list of dummy ChemicalComponent objects. These sequences don't mean anything
    """
    return [SequenceComponent(seq="KVFGRCELAAAMKRHGLD"), SequenceComponent(seq="QATNRNTDGSTDYGIL")]
