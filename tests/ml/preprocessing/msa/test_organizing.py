"""Tests for atomworks.ml.preprocessing.msa.organizing.

Covers organize_msas's contract with the lookup side (get_msa_path /
finding.find_msas): anything organize_msas produces must be discoverable by
re-hashing the sequence, using the *matching* directory_depth.
"""

from pathlib import Path

import pytest

from atomworks.io.utils.io_utils import build_sharding_pattern
from atomworks.ml.preprocessing.msa.finding import find_paired_msas, find_template_alignments
from atomworks.ml.preprocessing.msa.organizing import (
    MSAOrganizationConfig,
    organize_msas,
    organize_paired_msas,
    organize_template_alignments,
)
from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path
from atomworks.ml.utils.misc import get_complex_id, hash_sequence

QUERY_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
SECOND_SEQUENCE = "MSEQVENCETWOFORPAIRINGTESTSXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def _fake_a3m(sequence: str) -> str:
    """A minimal, well-formed single-hit a3m block for `sequence`."""
    return f">query\n{sequence}\n>hit_1\n{sequence}\n"


def test_organize_then_find_msa_path(tmp_path: Path) -> None:
    """A raw a3m file, once organized, is discoverable via get_msa_path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "0.a3m").write_text(_fake_a3m(QUERY_SEQUENCE))
    directory_depth = 1

    # Derive the sharding_pattern from the same helper get_msa_path uses internally
    # (build_sharding_pattern(depth, chars_per_dir=2)), rather than hand-formatting it --
    # "/0:{depth}/" looks plausible but is wrong: depth is a *level* count with 2
    # chars/level baked in, not a character count, and get_msa_path silently returns
    # None (not an error) on a mismatch.
    organized_dir = tmp_path / "organized"
    organize_msas(
        raw_dir,
        organized_dir,
        MSAOrganizationConfig(
            input_extension=".a3m",
            output_extension=".a3m",
            sharding_pattern=build_sharding_pattern(depth=directory_depth),
        ),
    )

    msa_dir = {"dir": str(organized_dir), "extension": ".a3m", "directory_depth": directory_depth}
    found = get_msa_path(QUERY_SEQUENCE, [msa_dir])

    expected_hash = hash_sequence(QUERY_SEQUENCE)
    expected_path = organized_dir / expected_hash[:2] / f"{expected_hash}.a3m"
    assert found == expected_path
    assert found.read_text() == _fake_a3m(QUERY_SEQUENCE)


@pytest.mark.parametrize("dir_format", ["dict", "plain_path"])
def test_organize_paired_then_find_paired_msas(tmp_path: Path, dir_format: str) -> None:
    """Raw per-chain a3m files from one complex, once organized, are discoverable via
    find_paired_msas -- complex_id is derived from the files' own content (their
    first sequences, hashed and joined via get_complex_id), not passed in.

    Both directory formats must default to the writer's layout (complex_id sharded one
    level deep). A plain path must not auto-detect depth: the per-complex directory
    would be miscounted as a second shard level.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "0.a3m").write_text(_fake_a3m(QUERY_SEQUENCE))
    (raw_dir / "1.a3m").write_text(_fake_a3m(SECOND_SEQUENCE))

    organized_dir = tmp_path / "organized"
    returned_complex_id = organize_paired_msas(
        raw_dir, organized_dir, MSAOrganizationConfig(input_extension=".a3m", output_extension=".a3m")
    )

    expected_complex_id = get_complex_id([QUERY_SEQUENCE, SECOND_SEQUENCE])
    assert returned_complex_id == expected_complex_id

    msa_dir = {"dir": str(organized_dir), "extension": ".a3m"} if dir_format == "dict" else organized_dir
    missing, found = find_paired_msas([QUERY_SEQUENCE, SECOND_SEQUENCE], [msa_dir])

    assert missing == []
    for seq in (QUERY_SEQUENCE, SECOND_SEQUENCE):
        expected_path = organized_dir / expected_complex_id[:2] / expected_complex_id / f"{hash_sequence(seq)}.a3m"
        assert found[seq] == expected_path
        assert expected_path.read_text() == _fake_a3m(seq)


def test_organize_paired_with_repeated_chains_found_by_unique_set(tmp_path: Path) -> None:
    """A complex submitted with repeated chains (e.g. H2L2 -> one raw file per submitted
    chain) is organized under the same complex_id as its unique set, so it's found by
    looking up the unique sequences.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i, seq in enumerate([QUERY_SEQUENCE, QUERY_SEQUENCE, SECOND_SEQUENCE, SECOND_SEQUENCE]):
        (raw_dir / f"{i}.a3m").write_text(_fake_a3m(seq))

    organized_dir = tmp_path / "organized"
    complex_id = organize_paired_msas(
        raw_dir, organized_dir, MSAOrganizationConfig(input_extension=".a3m", output_extension=".a3m")
    )

    assert complex_id == get_complex_id([QUERY_SEQUENCE, SECOND_SEQUENCE])
    missing, found = find_paired_msas([QUERY_SEQUENCE, SECOND_SEQUENCE], [organized_dir])
    assert missing == []
    assert set(found) == {QUERY_SEQUENCE, SECOND_SEQUENCE}


def test_organize_paired_refuses_existing_complex_dir(tmp_path: Path) -> None:
    """Re-organizing a complex whose directory already exists (e.g. partial, from an
    interrupted run) raises rather than mixing chains from two separate searches.
    """
    config = MSAOrganizationConfig(input_extension=".a3m", output_extension=".a3m")
    organized_dir = tmp_path / "organized"

    first_run = tmp_path / "first_run"
    first_run.mkdir()
    (first_run / "0.a3m").write_text(_fake_a3m(QUERY_SEQUENCE))
    (first_run / "1.a3m").write_text(_fake_a3m(SECOND_SEQUENCE))
    complex_id = organize_paired_msas(first_run, organized_dir, config)

    retry = tmp_path / "retry"
    retry.mkdir()
    (retry / "0.a3m").write_text(_fake_a3m(QUERY_SEQUENCE) + ">hit_from_retry\nXXXX\n")
    (retry / "1.a3m").write_text(_fake_a3m(SECOND_SEQUENCE) + ">hit_from_retry\nXXXX\n")
    with pytest.raises(FileExistsError):
        organize_paired_msas(retry, organized_dir, config)

    complex_dir = organized_dir / complex_id[:2] / complex_id
    for seq in (QUERY_SEQUENCE, SECOND_SEQUENCE):
        assert (complex_dir / f"{hash_sequence(seq)}.a3m").read_text() == _fake_a3m(seq)


def _fake_m8_hit(sequence: str) -> str:
    """A minimal, well-formed single-row pdb70.m8 hit block for `sequence`."""
    return f"101\t1abc_A\t0.99\t{len(sequence)}\t0\t0\t1\t{len(sequence)}\t1\t{len(sequence)}\t1e-20\t100\n"


def test_organize_then_find_template_alignments(tmp_path: Path) -> None:
    """Raw pdb70.m8 hit blocks, once organized, are discoverable via
    find_template_alignments.
    """
    organized_dir = tmp_path / "templates"
    organize_template_alignments(
        {QUERY_SEQUENCE: _fake_m8_hit(QUERY_SEQUENCE), SECOND_SEQUENCE: ""},
        organized_dir,
    )

    template_dir = {"dir": str(organized_dir), "extension": ".m8", "directory_depth": 1}
    missing, found = find_template_alignments([QUERY_SEQUENCE, SECOND_SEQUENCE], [template_dir])

    assert missing == [SECOND_SEQUENCE]  # empty content -- no file written
    assert list(found) == [QUERY_SEQUENCE]

    expected_hash = hash_sequence(QUERY_SEQUENCE)
    expected_path = organized_dir / expected_hash[:2] / f"{expected_hash}.m8"
    assert found[QUERY_SEQUENCE] == expected_path
    assert expected_path.read_text() == _fake_m8_hit(QUERY_SEQUENCE)


def test_find_template_alignments_with_no_dirs_reports_all_missing() -> None:
    """No directories (the default) -- every sequence is reported missing, matching
    find_msas's behavior for an empty/unset search path.
    """
    missing, found = find_template_alignments([QUERY_SEQUENCE])
    assert missing == [QUERY_SEQUENCE]
    assert found == {}
