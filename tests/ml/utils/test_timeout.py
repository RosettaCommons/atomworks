import time

import pytest

from atomworks.common import timeout as common_timeout
from atomworks.ml.utils.timer import timeout


@pytest.mark.parametrize("timeout_decorator", [timeout, common_timeout])
def test_timeout_wrapper_no_timeout(timeout_decorator):
    payload = b"x" * (1024 * 1024)

    @timeout_decorator(timeout=2)
    def fast_function():
        return payload

    assert fast_function() == payload


def test_timeout_wrapper_with_timeout():
    @timeout(timeout=0.1)
    def slow_function():
        time.sleep(0.5)
        return "This should not be returned"

    with pytest.raises(TimeoutError):
        slow_function()


def test_timeout_wrapper_disable_timeout():
    @timeout(timeout=None)
    def slow_function():
        time.sleep(0.5)
        return "Success"

    assert slow_function() == "Success"


def test_timeout_wrapper_signal_strategy():
    @timeout(timeout=0.1, strategy="signal")
    def slow_function():
        time.sleep(0.5)

    with pytest.raises(TimeoutError):
        slow_function()


def test_timeout_wrapper_subprocess_strategy():
    @timeout(timeout=0.1, strategy="subprocess")
    def slow_function():
        time.sleep(0.5)

    with pytest.raises(TimeoutError):
        slow_function()


def test_timeout_wrapper_invalid_strategy():
    with pytest.raises(ValueError):

        @timeout(timeout=1, strategy="invalid")
        def function():
            pass


# Try timeout on RDKit
def test_timeout_on_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    hem_smiles = "Cc1c2n3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe]36[N]7=C(C=C8N6C(=C5)C(=C8C)C=C)C(=C(C7=C2)C)C=C)C)CCC(=O)O"
    mol = Chem.MolFromSmiles(hem_smiles)

    @timeout(timeout=0.5, strategy="subprocess")
    def generate_conformers(mol):
        # ... takes about 2 seconds to run
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
        return mol

    start_time = time.time()
    with pytest.raises(TimeoutError):
        generate_conformers(mol)
    end_time = time.time()
    assert end_time - start_time < 1.5  # More tha 0.5 since suprocesses must spawn, run, and communicate back


if __name__ == "__main__":
    pytest.main(["-v", __file__])
