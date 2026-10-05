"""Tests for atomworks.ml.preprocessing.msa.template_structures.

Does not hit the real RCSB API -- biotite.database.rcsb.fetch is monkeypatched.
"""

from pathlib import Path

import pytest
from biotite.database import RequestError

from atomworks.ml.preprocessing.msa import template_structures

# Mirrors the real pdb70.m8 column layout: M, pdb_chain, seq_id, aln_len, mismatches,
# gaps, q_start, q_end, t_start, t_end, e_value, bit_score, cigar.
M8_CONTENT = (
    "101\t1qfe_A\t1.000\t252\t0\t0\t1\t252\t1\t252\t1.748E-77\t260\t252M\n"
    "101\t1qfe_B\t1.000\t252\t0\t0\t1\t252\t1\t252\t1.748E-77\t260\t252M\n"  # same entry, different chain
    "101\t3oex_B\t0.769\t252\t58\t0\t1\t252\t2\t253\t6.149E-77\t258\t252M\n"
)

# "0000" and its error text match what biotite.database.rcsb.fetch raised against the
# real RCSB API when tested on 28-Aug-2026.
INVALID_PDB_ID = "0000"


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace RCSB fetching with a local fake; returns the list of fetched IDs."""
    calls = []

    def _fake_fetch(pdb_ids: str, format: str, target_path: str) -> str:
        calls.append(pdb_ids)
        if pdb_ids == INVALID_PDB_ID:
            raise RequestError(f"PDB ID {pdb_ids} is invalid")
        dest = Path(target_path) / f"{pdb_ids}.{format}"
        dest.write_text(f"fake mmCIF for {pdb_ids}")
        return str(dest)

    monkeypatch.setattr(template_structures, "fetch", _fake_fetch)
    return calls


@pytest.mark.parametrize("m8_content", [M8_CONTENT, "\n\n" + M8_CONTENT + "\n"], ids=["plain", "blank_lines"])
def test_parse_template_pdb_ids(m8_content: str) -> None:
    """One lowercased entry ID per hit row, in the server's best-first order."""
    assert template_structures.parse_template_pdb_ids(m8_content) == ["1qfe", "1qfe", "3oex"]


@pytest.mark.parametrize("empty_content", ["", "\n\n\n"])
def test_parse_template_pdb_ids_raises_on_empty_content(empty_content: str) -> None:
    """A sequence with no PDB70 hits never gets a .m8 file written (see
    organize_template_alignments), so empty content means a caller bug.
    """
    with pytest.raises(ValueError, match="No template hits found"):
        template_structures.parse_template_pdb_ids(empty_content)


@pytest.mark.parametrize(
    "max_hits, mirror_ids, expected_ids, expected_downloads",
    [
        (20, [], ["1qfe", "3oex"], ["1qfe", "3oex"]),
        (1, [], ["1qfe"], ["1qfe"]),
        (20, ["1qfe"], ["1qfe", "3oex"], ["3oex"]),
    ],
    ids=["dedupes_by_entry", "respects_max_hits", "uses_pdb_mirror"],
)
def test_fetch_template_structures(
    tmp_path: Path,
    fake_fetch: list[str],
    max_hits: int,
    mirror_ids: list[str],
    expected_ids: list[str],
    expected_downloads: list[str],
) -> None:
    """One structure per unique entry (1qfe_A and 1qfe_B share one mmCIF), best-first,
    capped at `max_hits` unique entries; entries already in the PDB mirror are used in
    place and not downloaded.
    """
    mirror_dir = tmp_path / "mirror"
    for pdb_id in mirror_ids:
        mirror_file = mirror_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz"
        mirror_file.parent.mkdir(parents=True)
        mirror_file.write_text("mirror mmCIF")

    output_dir = tmp_path / "downloads"
    pdb_ids = template_structures.parse_template_pdb_ids(M8_CONTENT)
    fetched = template_structures.fetch_template_structures(
        pdb_ids, output_dir, max_hits=max_hits, pdb_mirror_path=mirror_dir
    )

    assert fake_fetch == expected_downloads
    assert list(fetched) == expected_ids
    for pdb_id in expected_ids:
        if pdb_id in mirror_ids:
            assert fetched[pdb_id] == mirror_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz"
        else:
            assert fetched[pdb_id].read_text() == f"fake mmCIF for {pdb_id}"


def test_fetch_template_structures_from_m8_file_skips_failed_fetch(
    tmp_path: Path, fake_fetch: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """End to end from a .m8 file: a hit that fails to fetch (e.g. an obsolete/invalid
    PDB ID) is omitted with a warning, without failing the other hits.
    """
    m8_path = tmp_path / "template.m8"
    m8_path.write_text(M8_CONTENT + f"101\t{INVALID_PDB_ID}_A\t0.50\t10\t0\t0\t1\t10\t1\t10\t1e-5\t20\t10M\n")

    with caplog.at_level("WARNING", logger=template_structures.logger.name):
        fetched = template_structures.fetch_template_structures_from_m8_file(
            m8_path, tmp_path / "structures", pdb_mirror_path=None
        )

    assert set(fetched) == {"1qfe", "3oex"}
    assert len(caplog.records) == 1
    assert INVALID_PDB_ID in caplog.records[0].message
