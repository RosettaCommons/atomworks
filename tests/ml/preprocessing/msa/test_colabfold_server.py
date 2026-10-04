"""Tests for atomworks.ml.preprocessing.msa.colabfold_server.

Does not hit the real ColabFold MSA server -- query_colabfold_msa_server is
monkeypatched. See test_organizing.py for coverage of the underlying
organize_msas / get_msa_path contract this module relies on.
"""

import io
import tarfile
from pathlib import Path

import pytest

from atomworks.ml.preprocessing.msa import colabfold_server
from atomworks.ml.preprocessing.msa.colabfold_server import make_msas_colabfold_server_batch
from atomworks.ml.preprocessing.msa.finding import find_paired_msas, find_template_alignments, get_paired_msa_path
from atomworks.ml.transforms.msa._msa_loading_utils import get_msa_path
from atomworks.ml.utils.misc import get_complex_id, hash_sequence

QUERY_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
SECOND_SEQUENCE = "MSEQVENCETWOFORPAIRINGTESTSXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def _fake_a3m(sequence: str) -> str:
    """A minimal, well-formed single-hit a3m block for `sequence`."""
    return f">query\n{sequence}\n>hit_1\n{sequence}\n"


def _fake_query_colabfold_msa_server(x, prefix, user_agent, use_pairing=False, use_templates=False, **kwargs):
    # Mirrors the real function's contract: one a3m string per input sequence, in
    # input order. Does not touch the network or `prefix`. When use_templates is set,
    # also returns one raw template-alignment hit block per sequence (None for the
    # second sequence, mirroring a sequence with no PDB70 hits).
    a3m_lines = [_fake_a3m(seq) for seq in x]
    if use_templates:
        template_alignments = [_fake_m8_hit(seq) if i == 0 else None for i, seq in enumerate(x)]
        return a3m_lines, template_alignments
    return a3m_lines


def _fake_m8_hit(sequence: str) -> str:
    """A minimal, well-formed single-row pdb70.m8 hit block for `sequence`."""
    return f"101\t1abc_A\t0.99\t{len(sequence)}\t0\t0\t1\t{len(sequence)}\t1\t{len(sequence)}\t1e-20\t100\n"


def test_make_msas_colabfold_server_batch_writes_unpaired_and_paired_under_one_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One output_dir covers both unpaired and paired output -- the caller never
    constructs unpaired/paired paths itself, and never computes complex_id itself
    either (derived automatically from the a3m results' content). complex_id is
    sharded the same way sequence hashes are (many complexes in a large run); chain
    files within one complex's own directory stay flat (few chains per complex).
    """
    sequences = [QUERY_SEQUENCE, SECOND_SEQUENCE]
    complex_id = get_complex_id(sequences)

    monkeypatch.setattr(colabfold_server, "query_colabfold_msa_server", _fake_query_colabfold_msa_server)

    output_dir = tmp_path / "msas"
    make_msas_colabfold_server_batch(sequences, output_dir, complexes=[sequences], user_agent="test-agent")

    unpaired_dir = {"dir": str(output_dir / "unpaired"), "extension": ".a3m.gz", "directory_depth": 1}
    for seq in sequences:
        found = get_msa_path(seq, [unpaired_dir])
        assert found is not None, f"unpaired MSA not found for sequence hash {hash_sequence(seq)}"

    paired_dir = {"dir": str(output_dir / "paired"), "extension": ".a3m.gz"}
    for seq in sequences:
        found = get_paired_msa_path(complex_id, seq, [paired_dir])
        assert found is not None, f"paired MSA not found for sequence hash {hash_sequence(seq)}"
        # complex_id is sharded (not a flat <paired>/<complex_id>/... directory)
        assert found.parent.parent.name == complex_id[:2]


def test_make_msas_colabfold_server_batch_skips_empty_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No sequences and no complexes -- no query_colabfold_msa_server call at all."""
    calls = []
    monkeypatch.setattr(
        colabfold_server,
        "query_colabfold_msa_server",
        lambda *a, **kw: calls.append(1) or _fake_query_colabfold_msa_server(*a, **kw),
    )

    make_msas_colabfold_server_batch([], tmp_path / "msas", complexes=None, user_agent="test-agent")

    assert calls == []


def test_make_msas_colabfold_server_batch_dedupes_complexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Complexes with the same complex_id (reordered, or with repeated chains) are
    submitted as one paired query, rather than re-querying and then failing on the
    existing complex directory.
    """
    paired_calls = []

    def fake(x, *args, use_pairing=False, **kwargs):
        if use_pairing:
            paired_calls.append(x)
        return _fake_query_colabfold_msa_server(x, *args, use_pairing=use_pairing, **kwargs)

    monkeypatch.setattr(colabfold_server, "query_colabfold_msa_server", fake)

    complexes = [
        [QUERY_SEQUENCE, SECOND_SEQUENCE],
        [SECOND_SEQUENCE, QUERY_SEQUENCE],
        [QUERY_SEQUENCE, QUERY_SEQUENCE, SECOND_SEQUENCE, SECOND_SEQUENCE],
    ]
    output_dir = tmp_path / "msas"
    make_msas_colabfold_server_batch([], output_dir, complexes=complexes, user_agent="test-agent")

    assert len(paired_calls) == 1
    missing, _ = find_paired_msas([QUERY_SEQUENCE, SECOND_SEQUENCE], [output_dir / "paired"])
    assert missing == []


def test_make_msas_colabfold_server_batch_organizes_template_alignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_templates=True organizes raw template-alignment hits for the unpaired
    batch, discoverable via find_template_alignments, keyed by the same sequence
    hash as the unpaired MSAs. A sequence with no hits (SECOND_SEQUENCE here) yields
    no file, and is reported missing.
    """
    sequences = [QUERY_SEQUENCE, SECOND_SEQUENCE]
    monkeypatch.setattr(colabfold_server, "query_colabfold_msa_server", _fake_query_colabfold_msa_server)

    output_dir = tmp_path / "msas"
    make_msas_colabfold_server_batch(sequences, output_dir, use_templates=True, user_agent="test-agent")

    template_dir = {"dir": str(output_dir / "templates"), "extension": ".m8", "directory_depth": 1}
    missing, found = find_template_alignments(sequences, [template_dir])

    assert missing == [SECOND_SEQUENCE]
    assert found[QUERY_SEQUENCE].read_text() == _fake_m8_hit(QUERY_SEQUENCE)


def test_make_msas_colabfold_server_batch_auto_includes_complex_members_in_unpaired_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complex members not explicitly passed in `sequences` are still submitted as an
    unpaired query -- every chain needs its own unpaired MSA (and, if requested,
    template alignment) regardless of whether it also participates in pairing.
    """
    monkeypatch.setattr(colabfold_server, "query_colabfold_msa_server", _fake_query_colabfold_msa_server)

    output_dir = tmp_path / "msas"
    make_msas_colabfold_server_batch(
        [], output_dir, complexes=[[QUERY_SEQUENCE, SECOND_SEQUENCE]], use_templates=True, user_agent="test-agent"
    )

    unpaired_dir = {"dir": str(output_dir / "unpaired"), "extension": ".a3m.gz", "directory_depth": 1}
    for seq in (QUERY_SEQUENCE, SECOND_SEQUENCE):
        assert get_msa_path(seq, [unpaired_dir]) is not None, f"unpaired MSA not found for {seq[:10]}..."

    template_dir = {"dir": str(output_dir / "templates"), "extension": ".m8", "directory_depth": 1}
    missing, found = find_template_alignments([QUERY_SEQUENCE, SECOND_SEQUENCE], [template_dir])
    assert missing == [SECOND_SEQUENCE]
    assert found[QUERY_SEQUENCE].read_text() == _fake_m8_hit(QUERY_SEQUENCE)


def test_make_msas_colabfold_server_batch_paired_submission_never_requests_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paired/complex query itself never requests templates (not supported by
    the ColabFold server) -- only the (now auto-included) unpaired call does.
    """
    calls = []

    def _tracking_fake(x, prefix, user_agent, use_pairing=False, use_templates=False, **kwargs):
        calls.append({"use_pairing": use_pairing, "use_templates": use_templates})
        return _fake_query_colabfold_msa_server(
            x, prefix, user_agent, use_pairing=use_pairing, use_templates=use_templates, **kwargs
        )

    monkeypatch.setattr(colabfold_server, "query_colabfold_msa_server", _tracking_fake)

    output_dir = tmp_path / "msas"
    make_msas_colabfold_server_batch(
        [], output_dir, complexes=[[QUERY_SEQUENCE, SECOND_SEQUENCE]], use_templates=True, user_agent="test-agent"
    )

    paired_calls = [c for c in calls if c["use_pairing"]]
    unpaired_calls = [c for c in calls if not c["use_pairing"]]
    assert len(paired_calls) == 1 and paired_calls[0]["use_templates"] is False
    assert len(unpaired_calls) == 1 and unpaired_calls[0]["use_templates"] is True


class _FakeResponse:
    """Minimal stand-in for requests.Response -- only what query_colabfold_msa_server
    actually reads (.json() for submit/status, .content for download)."""

    def __init__(self, json_data: dict | None = None, content: bytes = b"") -> None:
        self._json_data = json_data
        self.content = content

    def json(self) -> dict:
        return self._json_data


def _build_tar_gz(files: dict[str, str]) -> bytes:
    """Build an in-memory tar.gz matching the layout of ColabFold's out.tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_query_colabfold_msa_server_parses_raw_template_hits_without_fetching_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_templates=True on the real query_colabfold_msa_server (only the HTTP layer
    is mocked, not the raw output directory -- see
    test_query_colabfold_msa_server_raises_if_raw_dir_exists for why) returns the raw
    pdb70.m8 hit rows per sequence, without any request to fetch the underlying PDB70
    structure database -- that retrieval step isn't implemented.
    """
    tar_gz_bytes = _build_tar_gz(
        {
            "uniref.a3m": f">101\n{QUERY_SEQUENCE}\n>hit_1\n{QUERY_SEQUENCE}\n",
            "bfd.mgnify30.metaeuk30.smag30.a3m": f">101\n{QUERY_SEQUENCE}\n",
            # M=102 belongs to a different query in the same batch job -- shouldn't
            # leak into this query's returned template alignment.
            "pdb70.m8": _fake_m8_hit(QUERY_SEQUENCE) + "102\t2xyz_B\t0.5\t1\t0\t0\t1\t1\t1\t1\t1e-5\t50\n",
        }
    )

    def _fake_post(url: str, **kwargs) -> _FakeResponse:
        assert url.endswith("/ticket/msa")
        return _FakeResponse(json_data={"status": "COMPLETE", "id": "job123"})

    def _fake_get(url: str, **kwargs) -> _FakeResponse:
        assert url.endswith("/result/download/job123")
        return _FakeResponse(content=tar_gz_bytes)

    monkeypatch.setattr(colabfold_server.requests, "post", _fake_post)
    monkeypatch.setattr(colabfold_server.requests, "get", _fake_get)

    prefix = tmp_path / "raw"  # must not already exist -- see the guard test below
    a3m_lines, template_alignments = colabfold_server.query_colabfold_msa_server(
        [QUERY_SEQUENCE], prefix=prefix, user_agent="test-agent", use_templates=True
    )

    # Gathered from both uniref.a3m and the env (bfd) file, concatenated under the
    # shared M=101 key -- mirrors the real server output, where both are read in.
    assert a3m_lines == [f">101\n{QUERY_SEQUENCE}\n>hit_1\n{QUERY_SEQUENCE}\n>101\n{QUERY_SEQUENCE}\n"]
    assert template_alignments == [_fake_m8_hit(QUERY_SEQUENCE)]
    # No structure database was downloaded -- no templates_* directory created.
    assert not list(prefix.glob("templates_*"))


def test_query_colabfold_msa_server_raises_if_raw_dir_exists(tmp_path: Path) -> None:
    """A pre-existing raw output directory is always treated as a stale leftover from
    an earlier (likely failed) run at this exact path -- reusing it would risk
    silently parsing a *different* query's out.tar.gz, corrupting results in a way
    that's very hard to debug downstream.

    See https://github.com/aqlaboratory/openfold-3/issues/39, where this exact
    footgun (a stale out.tar.gz silently reused via `if not os.path.isfile(...)`)
    was hit in production and fixed by erroring instead of reusing.
    """
    prefix = tmp_path / "raw"
    prefix.mkdir()

    with pytest.raises(FileExistsError, match="raw output directory"):
        colabfold_server.query_colabfold_msa_server([QUERY_SEQUENCE], prefix=prefix, user_agent="test-agent")
