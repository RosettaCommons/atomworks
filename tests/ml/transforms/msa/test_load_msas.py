import copy
import logging
import time
from typing import Any

import numpy as np
import pytest

from atomworks.enums import ChainType
from atomworks.ml.transforms.msa._msa_constants import (
    AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
    RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
)
from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path
from atomworks.ml.transforms.msa.msa import LoadPolymerMSAs
from atomworks.ml.utils.testing import cached_parse
from tests.ml.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MSA_TEST_CASES = [
    {
        # Protein
        "pdb_id": "5gam",
        "chain_id": "C",
        "sequence": "MEGDDLFDEFGNLIGVDPFDSDEEESVLDEQEQYQTNTFEGSGNNNEIESRQLTSLGSKKELGISLEHPYGKEVEVLMETKNTQSPQTPLVEPVTERTKLQEHTIFTQLKKNIPKTRYNRDYMLSMANIPERIINVGVIGPLHSGKTSLMDLLVIDSHKRIPDMSKNVELGWKPLRYLDNLKQEIDRGLSIKLNGSTLLCTDLESKSRMINFLDAPGHVNFMDETAVALAASDLVLIVIDVVEGVTFVVEQLIKQSIKNNVAMCFVINKLDRLILDLKLPPMDAYLKLNHIIANINSFTKGNVFSPIDNNIIFASTKLGFTFTIKEFVSYYYAHSIPSSKIDDFTTRLWGSVYYHKGNFRTKPFENVEKYPTFVEFILIPLYKIFSYALSMEKDKLKNLLRSNFRVNLSQEALQYDPQPFLKHVLQLIFRQQTGLVDAITRCYQPFELFDNKTAHLSIPGKSTPEGTLWAHVLKTVDYGGAEWSLVRIYSGLLKRGDTVRILDTSQSESRQKRQLHDISKTETSNEDEDSKTETPSCEVEEIGLLGGRYVYPVHEAHKGQIVLIKGISSAYIKSATLYSVKSKEDMKQLKFFKPLDYITEAVFKIVLQPLLPRELPKLLDALNKISKYYPGVIIKVEESGEHVILGNGELYMDCLLYDLRASYAKIEIKISDPLTVFSESCSNESFASIPVSNSISRLGEENLPGLSISVAAEPMDSKMIQDLSRNTLGKGQNCLDIDGIMDNPRKLSKILRTEYGWDSLASRNVWSFYNGNVLINDTLPDEISPELLSKYKEQIIQGFYWAVKEGPLAEEPIYGVQYKLLSISVPSDVNIDVMKSQIIPLMKKACYVGLLTAIPILLEPIYEVDITVHAPLLPIVEELMKKRRGSRIYKTIKVAGTPLLEVRGQVPVIESAGFETDLRLSTNGLGMCQLYFWHKIWRKVPGDVLDKDAFIPKLKPAPINSLSRDFVMKTRRRKGISTGGFMSNDGPTLEKYISAELYAQLRENGLVP",
        "min_sequences_in_msa": 1000,  # common protein, should have many sequences
        "spot_check": {
            "index": 1,
            "sequence": "--MDDLYDEFGQFLGFPQEFTSYEQSSEE----VQGEAAYSTLQG-DLDQE-------ATDVVLDANGNFDDDVEVLLEVED-REPDKPLVAGDL-------RPKGYDKCDKIPKAMFDREYLQSILAIPERQLNVGIFGPLHSGKTSFADMFALDTHHNLPSLTKKVKEGWLPFKYLDQERIEKERGVSLRLNGMTFGYESSRGRTYAVTMLDTPGHVNFWDDVGITLTCCQYGIVVIDVAEGVTSVVLKLFKELEQNGIEFIVVLNKIDRLALDLRLPADAAYWRLLHIVEQVNRHTKE-TFSPELGNVLFSSTKFGFVFSIESFVNSFYAKSLKDK-TEQFVARLWGLINYWDGEFNETEF--ISERNSFFVFILQPLYKVITHGLSASAEELQRVIKDNFQVNLSDETLSKDPQPLLFSIFRSIFPHHHCVIDSISRLRDRSFDISA-----------ND-GETLVHVLRHIKVNGTNWSLCRIAQGSLITGRKLYIFNESVDSIVDHAD--------------D---EYPKITIERIALMGGRYAYEVKEAQQGQLVLLKGFEDEFTKFATLS-------STVRNPLPPINYLNESVFKFAIQPQKPSDLPRLLHGLQLANGFYPSLVVRVEESGENIIVGTGELYLDCVMDELRKTFCEIEIKISQPLVQITESCNSESFASIPVKSNNGI--------VSISVMAEKLDDKIVHDLTHGEIN--------LSELNNVRKFSKRLRTEYGWDSLAARNFWGLSQCNVFVDDTLPDETDKKLLKRYKEYILQGFEWAVKEGPLADERMHACQFKLLELKVQEDKIDEFIPSQLVPLTRKACYIALMTAAPIVMEPIYEVDIV-------------------------NVQGTPFTEIKAQLPVIESIGFETDLRTATIGKGMCQMHFWNKIWRRVPGDVLDEEAFIPKLKPAPAASLSRDFVVKTRRRKGLSESGHMTQDGPSLKNYIDDELFEKLKQKGLV-",
            "num_insertions": 2,
            "tax_id": "1427455",
        },
    },
    {
        # RNA
        "pdb_id": "5gam",
        "chain_id": "A",
        "sequence": "AAGCAGCUUUACAGAUCAAUGGCGGAGGGAGGUCAACAUCAAGAACUGUGGGCCUUUUAUUGCCUAUAGAACUUAUAACGAACAUGGUUCUUGCCUUUUACCAGAACCAUCCGGGUGUUGUCUCCAUAGAAACAGGUAAAGCUGUCCGUUACUGUGGGCUUGCCAUAUUUUUUGGAAC",
        "min_sequences_in_msa": 10,  # NA MSAs are much shorter than protein MSAs
        "spot_check": {
            "index": 1,
            "sequence": "AAGCAGCUUGACAGAUCAAUGGCGGAGGGAGGUCAACAUCAAGAACUGUGGGACUUUUAUUGCCUAUAGAACUUAUAACGAACAUGGUUCUUGCCUUUUACCAGAACCAUCCGGGUGUUGUCUCCAUAGAAACAGGUAAAGCUGUCCGUUACUGCCAGCUUGCCAUAUUUUUUGGAA-",
            "num_insertions": 0,
            "tax_id": "",
        },
    },
]


def _encode_sequence(sequence: str, chain_type: ChainType):
    lookup_table = (
        AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE
        if chain_type.is_protein()
        else RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE
    )
    return lookup_table[np.frombuffer(sequence.encode(), dtype=np.int8)]


def _validate_msa_results(test_case: dict[str, Any], result: dict[str, Any], chain_type: str):
    """Perform assertions to validate the MSA results against the test case."""
    spot_check_index = test_case["spot_check"]["index"]

    assert result["msa"].shape[0] >= test_case["min_sequences_in_msa"], "MSA has too few sequences"
    assert np.all(result["msa"][0] == _encode_sequence(test_case["sequence"], chain_type)), (
        "Query sequence is incorrect"
    )

    # Spot check assertions
    assert np.all(
        result["msa"][spot_check_index] == _encode_sequence(test_case["spot_check"]["sequence"], chain_type)
    ), "Spot check sequence is incorrect"
    assert np.sum(result["ins"][spot_check_index]) == test_case["spot_check"]["num_insertions"], (
        "Incorrect number of insertions"
    )
    assert result["ins"][spot_check_index].shape[0] == len(result["msa"][spot_check_index]), "Incorrect insertion shape"
    assert result["tax_ids"][spot_check_index].item() == str(test_case["spot_check"]["tax_id"]), "Incorrect tax ID"

    # Sequence similarity sanity checks
    calculated_sequence_similarity = np.mean(result["msa"][spot_check_index] == result["msa"][0])
    assert calculated_sequence_similarity == result["sequence_similarity"][spot_check_index], (
        "Incorrect sequence similarity"
    )
    assert result["sequence_similarity"][0] == 1.0, "Query sequence should have 100% similarity with itself"


@pytest.fixture
def load_polymer_msas_transform():
    """Fixture to create a LoadPolymerMSAs transformation pipeline."""
    return LoadPolymerMSAs(
        protein_msa_dirs=PROTEIN_MSA_DIRS, rna_msa_dirs=RNA_MSA_DIRS, max_msa_sequences=2_000, msa_cache_dir=None
    )


@pytest.mark.parametrize("test_case", MSA_TEST_CASES)
def test_load_msas(test_case: dict[str, Any], load_polymer_msas_transform: LoadPolymerMSAs):
    """Test a series of hand-picked cases to ensure that the MSA loading pipeline is functioning correctly.

    We will check that the MSA has a minimum number of sequences, that the query sequence is correct, and that a spot check is correct.
    """
    data = cached_parse(test_case["pdb_id"], convert_mse_to_met=True)
    chain_type = data["chain_info"][test_case["chain_id"]]["chain_type"]
    output = load_polymer_msas_transform(data)

    result = output["polymer_msas_by_chain_id"][test_case["chain_id"]]

    _validate_msa_results(test_case, result, chain_type)


@pytest.mark.slow
@pytest.mark.parametrize("test_case", MSA_TEST_CASES)
def test_cache_msas(test_case: dict[str, Any], tmp_path: str, load_polymer_msas_transform):
    """Tests the MSA caching functionality by loading the same MSA with and without caching and comparing the results."""
    data = cached_parse(test_case["pdb_id"], convert_mse_to_met=True)

    # Load with caching turned off
    start_time = time.time()
    out_without_cache = load_polymer_msas_transform(copy.deepcopy(data))
    first_run_time = time.time() - start_time

    # Load with caching turned on
    cache_pipeline = LoadPolymerMSAs(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        max_msa_sequences=2_000,
        msa_cache_dir=tmp_path / "msa_cache",
    )

    # (Warmup, which caches the MSA)
    out_with_cache_1 = cache_pipeline(copy.deepcopy(data))

    # ... and again, loading from cache
    start_time = time.time()
    out_with_cache_2 = cache_pipeline(copy.deepcopy(data))
    last_run_time = time.time() - start_time

    # The results should be the same
    chain_id = test_case["chain_id"]
    for key in out_without_cache["polymer_msas_by_chain_id"][chain_id]:
        assert np.array_equal(
            out_without_cache["polymer_msas_by_chain_id"][chain_id][key],
            out_with_cache_1["polymer_msas_by_chain_id"][chain_id][key],
        )
        assert np.array_equal(
            out_with_cache_1["polymer_msas_by_chain_id"][chain_id][key],
            out_with_cache_2["polymer_msas_by_chain_id"][chain_id][key],
        )

    # The second run should be faster than non-cached MSA loading
    assert last_run_time < first_run_time * 0.7, "Cached MSA loading should be >2x faster than non-cached"


def _check_coverage_for_pdb_id(
    pdb_id: str, protein_msa_dirs: list[str] | None = None, rna_msa_dirs: list[str] | None = None
):
    """Utility function to evaluate the MSA coverage for a single PDB ID."""
    data = cached_parse(pdb_id, convert_mse_to_met=True)

    example_n_proteins = example_n_proteins_with_msas = 0
    example_n_rna = example_n_rna_with_msa = 0

    # Count polymers
    chain_ids = np.unique(data["atom_array"].chain_id)
    for chain_id in chain_ids:
        if data["chain_info"][chain_id]["chain_type"].is_protein():
            # Skip peptides
            if len(data["chain_info"]["A"]["res_id"]) > 10:
                example_n_proteins += 1
        elif data["chain_info"][chain_id]["chain_type"] == ChainType.RNA:
            example_n_rna += 1

    # Load MSAs
    load_polymer_msas_transform = LoadPolymerMSAs(
        protein_msa_dirs=protein_msa_dirs, rna_msa_dirs=rna_msa_dirs, max_msa_sequences=2_000, msa_cache_dir=None
    )
    output = load_polymer_msas_transform(data)

    # Count MSAs
    for chain_id, msa in output["polymer_msas_by_chain_id"].items():
        if data["chain_info"][chain_id]["chain_type"].is_protein():
            if msa["msa"].shape[0] > 1 and msa["msa"].shape[1] > 10:
                example_n_proteins_with_msas += 1
        elif data["chain_info"][chain_id]["chain_type"] == ChainType.RNA and msa["msa"].shape[0] > 1:
            example_n_rna_with_msa += 1

    return {
        "n_proteins": example_n_proteins,
        "n_proteins_with_msas": example_n_proteins_with_msas,
        "n_rna": example_n_rna,
        "n_rna_with_msa": example_n_rna_with_msa,
    }


def test_msas_with_mse():
    """Check that we correctly find MSAs for proteins with selenomethisone residues."""
    results = _check_coverage_for_pdb_id("7dsu", PROTEIN_MSA_DIRS, RNA_MSA_DIRS)
    assert results["n_proteins_with_msas"] == results["n_proteins"], (
        "All proteins should have MSAs after MSE conversion"
    )


@pytest.mark.parametrize("test_case", MSA_TEST_CASES)
def test_inference_msa_transform(test_case):
    """Test the LoadPolymerMSAsInference transformation pipeline, where we provide MSAs through the `chain_info` field"""
    data = cached_parse(test_case["pdb_id"], convert_mse_to_met=True)
    chain_id = test_case["chain_id"]
    chain_type = data["chain_info"][chain_id]["chain_type"]

    # ... spoof the MSA path in the chain info
    if chain_type.is_protein():
        sequence = data["chain_info"][chain_id]["processed_entity_non_canonical_sequence"]
        data["chain_info"][chain_id]["msa_path"] = get_msa_path(sequence, PROTEIN_MSA_DIRS)
    elif chain_type == ChainType.RNA:
        # HACK: Replace U with T to match the RNA MSA file names (legacy issue)
        sequence = data["chain_info"][chain_id]["processed_entity_non_canonical_sequence"].replace("U", "T")
        data["chain_info"][chain_id]["msa_path"] = get_msa_path(sequence, RNA_MSA_DIRS)

    # Inference MSA pipeline
    inference_pipeline = LoadPolymerMSAs(
        protein_msa_dirs=[],  # No MSA directories
        rna_msa_dirs=[],
        max_msa_sequences=2_000,
        use_paths_in_chain_info=True,  # Use the paths in the chain info
    )

    inference_output = inference_pipeline(data)

    result = inference_output["polymer_msas_by_chain_id"][chain_id]

    _validate_msa_results(test_case, result, chain_type)


if __name__ == "__main__":
    pytest.main(["-s", "-v", "-m not very_slow", __file__])
