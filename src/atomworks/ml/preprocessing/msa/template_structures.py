"""Utility library for fetching template structures from RCSB, given m8 alignment files."""

import logging
from os import PathLike
from pathlib import Path

from biotite.database import RequestError
from biotite.database.rcsb import fetch

from atomworks.constants import PDB_MIRROR_PATH
from atomworks.io.utils.testing import get_pdb_path

logger = logging.getLogger(__name__)


def parse_template_pdb_ids(m8_content: str) -> list[str]:
    """Parse the lowercased PDB entry ID of each hit row in a raw ``pdb70.m8`` table.

    Rows are returned by the ColabFold MSA server already sorted best-first (ascending
    e-value); order is preserved, and an entry hit by several chains appears once per
    hit.

    Raises:
        ValueError: If `m8_content` contains no hit rows. A sequence with no PDB70 hits
            never gets a ``.m8`` file written, so this suggests an upstream error, e.g.
            an empty or wrong file was read.
    """
    # Column 2 is "<pdb_id>_<chain_id>", e.g. "1qfe_A".
    pdb_ids = [line.split("\t")[1].split("_", 1)[0].lower() for line in m8_content.splitlines() if line.strip()]

    if not pdb_ids:
        raise ValueError(
            "No template hits found in m8_content. A sequence with no PDB70 hits never gets a .m8 file "
            "written in the first place, so this indicates a bug -- e.g. an empty or wrong file was read."
        )

    return pdb_ids


def fetch_template_structures(
    pdb_ids: list[str],
    output_dir: PathLike,
    max_hits: int = 20,
    pdb_mirror_path: PathLike | None = PDB_MIRROR_PATH,
) -> dict[str, Path]:
    """Fetch mmCIF files from list of PDB ids.

    Each entry is first looked up in `pdb_mirror_path` via
    :py:func:`~atomworks.io.utils.testing.get_pdb_path`; only entries not found there are downloaded
    from RCSB into `output_dir`.

    Args:
        pdb_ids: List of PDB entry IDs to fetch.
        output_dir: Directory to download missing entries into, as ``<pdb_id>.cif``.
        max_hits: Maximum number of unique PDB entries to resolve.
        pdb_mirror_path: Local PDB mirror to check before downloading.

    Returns:
        Dict mapping each resolved PDB ID to its local mmCIF path (mirror or download).
        Entries that fail to download (e.g. obsolete/invalid IDs) are omitted with a
        logged warning.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    structure_paths: dict[str, Path] = {}
    for pdb_id in list(dict.fromkeys(p.lower() for p in pdb_ids))[:max_hits]:
        try:
            structure_paths[pdb_id] = Path(get_pdb_path(pdb_id, mirror_path=pdb_mirror_path))
            continue
        except FileNotFoundError:
            pass  # Not in the mirror (or no mirror given); download below.
        try:
            structure_paths[pdb_id] = Path(fetch(pdb_ids=pdb_id, format="cif", target_path=str(output_path)))
        except RequestError as e:
            logger.warning(f"Could not fetch template structure {pdb_id} from RCSB, skipping: {e}")

    n_mirror = sum(1 for path in structure_paths.values() if not path.is_relative_to(output_path))
    logger.info(f"Resolved {len(structure_paths)} template structures ({n_mirror} from the local PDB mirror)")
    return structure_paths


def fetch_template_structures_from_m8_file(
    m8_path: PathLike,
    output_dir: PathLike,
    max_hits: int = 20,
    pdb_mirror_path: PathLike | None = PDB_MIRROR_PATH,
) -> dict[str, Path]:
    """Parse a raw ``pdb70.m8`` file and resolve mmCIF files for its top hits.

    See :py:func:`parse_template_pdb_ids` and :py:func:`fetch_template_structures`.
    """
    pdb_ids = parse_template_pdb_ids(Path(m8_path).read_text())
    return fetch_template_structures(pdb_ids, output_dir, max_hits=max_hits, pdb_mirror_path=pdb_mirror_path)
