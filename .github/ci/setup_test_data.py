"""Install the checksum-pinned public test pack and record downloaded PDB inputs."""

import hashlib
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

CACHE = Path(".ci-data")
DATA = Path("tests/data")


def fetch_pdb(pdb_id: str) -> str:
    """Fetch missing structures over HTTPS; record bytes even when already present."""
    path = DATA / "pdb" / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.cif.gz", path)
    return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    digest, filename = Path(".github/ci/test_pack.sha256").read_text().split()
    archive = CACHE / filename
    if not archive.exists():
        urlretrieve(f"https://files.ipd.uw.edu/pub/atomworks/{filename}", archive)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"Test-pack checksum mismatch: expected {digest}, got {actual}")
    with tarfile.open(archive) as bundle:
        bundle.extractall(DATA, filter="data")
    ids = (DATA / "shared/test_pdb_ids.txt").read_text().lower().split()
    ids += Path(".github/ci/extra_pdb_ids.txt").read_text().split()
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(fetch_pdb, sorted(set(ids))))
    (CACHE / "pdb-sha256.txt").write_text("".join(hashes))
