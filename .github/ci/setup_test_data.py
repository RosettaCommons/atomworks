"""Install the checksum-pinned public test pack and record downloaded PDB inputs."""

import gzip
import hashlib
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, urlretrieve

CACHE = Path(".ci-data")
DATA = Path("tests/data")
PDB_VERSIONS = {
    pdb_id: (revision, digest)
    for pdb_id, revision, digest in (
        line.split() for line in Path(".github/ci/pdb_versions.tsv").read_text().splitlines()
    )
}


def fetch_pdb(pdb_id: str) -> str:
    """Pin stored-result inputs to their reviewed revisions; record every downloaded input."""
    path = DATA / "pdb" / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    revision, digest = PDB_VERSIONS.get(pdb_id, (None, None))
    data = path.read_bytes() if path.exists() else b""
    if not data or (digest and hashlib.sha256(gzip.decompress(data)).hexdigest() != digest):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        if revision:
            name = f"pdb_0000{pdb_id}"
            url = (
                f"https://files-versioned.wwpdb.org/pdb_versioned/data/entries/"
                f"{pdb_id[1:3]}/{name}/{name}_xyz_v{revision}.cif.gz"
            )
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if digest and hashlib.sha256(gzip.decompress(data)).hexdigest() != digest:
            raise ValueError(f"PDB {pdb_id} revision {revision} checksum mismatch")
        path.write_bytes(data)
    return f"{hashlib.sha256(data).hexdigest()}  {path}\n"


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
