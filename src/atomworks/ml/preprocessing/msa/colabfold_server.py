"""Generate MSAs by querying the public ColabFold MSA server over HTTP.

Unlike :py:mod:`~atomworks.ml.preprocessing.msa.generating`, which runs MMseqs2/HHblits
locally against on-disk databases, this module submits sequences to a remote MSA
server (by default ``https://api.colabfold.com``) and polls for results.

Adapted from ColabFold's ``run_mmseqs2``
(https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py#L69).

Can also fetch raw template-alignment hits (``pdb70.m8`` rows) alongside the MSA via
``use_templates``. To download the hit structures, see
:py:mod:`~atomworks.ml.preprocessing.msa.template_structures`.

References:
    * Mirdita, M. et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19, 679-682.
"""

import logging
import os
import random
import tarfile
import tempfile
import time
from os import PathLike
from pathlib import Path
from typing import Literal

import requests
from tqdm import tqdm

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.organizing import (
    MSAOrganizationConfig,
    organize_msas,
    organize_paired_msas,
    organize_template_alignments,
)
from atomworks.ml.utils.misc import get_complex_id

logger = logging.getLogger(__name__)


TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"


class ColabFoldServerResultError(RuntimeError):
    """Raised when a ColabFold MSA server download is missing expected outputs.

    The public ColabFold server can return the wrong cached job for a ticket --
    e.g. an unpaired MSA (no ``pair.a3m``) in response to a paired request -- which
    otherwise surfaces as an opaque ``FileNotFoundError`` downstream.
    """


def _validate_expected_msa_files(a3m_files: list[str], tar_gz_file: str, *, use_pairing: bool) -> None:
    """Verify the download produced every expected a3m file (present and non-empty).

    Guards against the ColabFold server returning an unexpected or incomplete result:
    instead of letting a later ``open()`` raise a bare ``FileNotFoundError``, raise a
    clear error naming the expected files, the missing ones, and what the downloaded
    tarball actually contained.

    Args:
        a3m_files: Absolute paths of the a3m files this query is expected to yield.
        tar_gz_file: Path of the downloaded ``out.tar.gz`` (read only for diagnostics).
        use_pairing: Whether this was a paired query (used only for the message).

    Raises:
        ColabFoldServerResultError: If any expected file is missing or empty.
    """
    missing: list[str] = [f for f in a3m_files if not os.path.isfile(f) or os.path.getsize(f) == 0]
    if not missing:
        return

    try:
        with tarfile.open(tar_gz_file) as tar_gz:
            members: list[str] = sorted(m.name for m in tar_gz.getmembers())
    except (tarfile.TarError, OSError):
        members = ["<missing or unreadable out.tar.gz>"]

    query_kind = "paired" if use_pairing else "unpaired"
    raise ColabFoldServerResultError(
        f"ColabFold {query_kind} MSA query returned an unexpected or incomplete "
        "result.\n"
        f"  expected (non-empty): {[os.path.basename(f) for f in a3m_files]}\n"
        f"  missing/empty:        {[os.path.basename(f) for f in missing]}\n"
        f"  tarball contained:    {members}\n"
        f"  download:             {tar_gz_file}\n"
        "The MSA server likely returned the wrong cached job for this query "
        "(e.g. an unpaired result for a paired request). Please inspect the tarball "
        "contents and re-run the query if necessary."
    )


def query_colabfold_msa_server(
    x: list[str],
    prefix: Path,
    user_agent: str,
    use_templates: bool = False,
    use_pairing: bool = False,
    pairing_strategy: Literal["greedy", "complete"] = "greedy",
    use_env: bool = True,
    use_filter: bool = True,
    filter: bool | None = None,
    host_url: str = "https://api.colabfold.com",
) -> list[str] | tuple[list[str], list[str]]:
    """Submits a single query to the ColabFold MSA server.

    Adapted from ColabFold's ``run_mmseqs2``:
    https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py#L69
    Upstream variable names (``ID``, ``REDO``, ``TIME``, ...) and the per-request
    retry loops are kept as-is to stay easy to diff against upstream; ruff's
    N803/N806 naming checks are disabled for this file in ``pyproject.toml``.

    Args:
        x (list[str]):
            List of amino acid sequences to query the MSA server with.
        prefix (Path):
            Output directory to save the results to. Must not already exist -- a
            pre-existing directory here is always treated as a stale leftover from an
            earlier run and rejected, rather than silently reused (see `Raises`).
        user_agent (str):
            User associated with API call.
        use_templates (bool, optional):
            Whether to return raw template-alignment hits (``pdb70.m8`` rows) for each
            query sequence. Defaults to False. If use_pairing is True, this internally
            gets set to False. Only the raw hit table is returned -- to download the
            hit structures, see :py:mod:`~atomworks.ml.preprocessing.msa.template_structures`.
        use_pairing (bool, optional):
            Whether to generate a single paired MSA. Defaults to False.
        pairing_strategy (str, optional):
            Pairing method, one of ["complete", "greedy"]. For pairing of more than 2
            chains, "complete" requires a taxonomic group to be present in all chains,
            where greedy requires it to be in only 2.
        use_env (bool, optional):
            Whether to align against env db (BFD/cfdb). Defaults to True.
        use_filter (bool, optional):
            Whether to apply diversity filter. Defaults to True.
        filter (bool | None, optional):
            Legacy option to enable diversity filter. Defaults to None.
        host_url (str, optional):
            Host url for MSA server. Defaults to "https://api.colabfold.com".

    Returns:
        list[str] | tuple[list[str], list[str | None]]:
            List of MSA strings in a3m format, one per query sequence. If use_templates
            is True, also returns a list of raw template-alignment hit blocks (tab-
            separated ``pdb70.m8`` rows, one per query sequence, in the same order as
            `x`); a sequence with no hits gets `None`.

    Raises:
        FileExistsError: If `prefix` already exists -- see the `prefix` arg.
    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Normalize the host: a trailing slash would produce a doubled slash (e.g.
    # ".../com//ticket/msa") in the f-strings below. The server 301-redirects the
    # doubled slash, and requests downgrades the POST to GET while dropping the body,
    # yielding a misleading "invalid ID".
    host_url = str(host_url).rstrip("/")

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent"
            "(e.g., 'toolname/version contact@email') to help"
            "us debug in case of problems. This warning will become an error"
            "in the future."
        )

    def submit(seqs: list[str], mode: str, N: int = 101) -> None:
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        error_count = 0
        while True:
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger
                # than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server.Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID: str) -> dict[str, str]:
        error_count = 0
        while True:
            try:
                res = requests.get(f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers)
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching status from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server.Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID: str, path: str) -> None:
        error_count = 0
        while True:
            try:
                # (connect_timeout, read_timeout): a single float applies to both, and
                # 6.02s is too tight for a read timeout on a results tarball that can be
                # several MB for a deep MSA -- keep the connect timeout tight but give the
                # actual download more room.
                res = requests.get(
                    f"{host_url}/result/download/{ID}",
                    timeout=(6.02, 60),
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while fetching result from MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server.Retrying... ({error_count}/5)")
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    seqs = [x] if isinstance(x, str) else x

    # Compatibility to old option
    if filter is not None:
        use_filter = filter

    # Setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"
    if use_pairing:
        use_templates = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # Put everything in the same dir. A directory that already exists here is always
    # leftover from an earlier (likely failed) run at this exact path -- reusing it
    # risks silently parsing a *different* query's stale out.tar.gz, corrupting
    # results in a way that's very hard to debug downstream. Fail loudly instead of
    # guessing whether it's safe to reuse; the caller can inspect/save it or delete
    # it before retrying. See https://github.com/aqlaboratory/openfold-3/issues/39.
    path = f"{prefix}"
    if os.path.isdir(path):
        raise FileExistsError(
            f"ColabFold raw output directory already exists: {path}\n"
            "This is likely left over from a previous failed run and may contain "
            "stale results for a different query. Please remove it (or move it "
            "aside first if you want to inspect/save the raw files) before retrying."
        )
    os.mkdir(path)

    # Call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # Deduplicate and keep track of order
    seqs_unique = []
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # Run query
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception("MMseqs2 API is undergoing maintenance.Please try again in a few minutes.")

                # Wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # Prepare list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path, filter="data")

    # Validate the download produced the expected outputs before any downstream
    # code blindly opens them.
    _validate_expected_msa_files(a3m_files, tar_gz_file, use_pairing=use_pairing)

    # Parse raw template-alignment hits (pdb70.m8 rows), grouped by ColabFold's
    # internal per-sequence index M. Only the raw hit table is returned here; hit
    # structures are fetched separately (see template_structures).
    if use_templates:
        template_hit_lines: dict[int, list[str]] = {}
        with open(f"{path}/pdb70.m8") as f:
            for line in f:
                if not line.strip():
                    continue
                M = int(line.split("\t", 1)[0])
                template_hit_lines.setdefault(M, []).append(line)

    # Gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file) as f:
            for line in f:
                if len(line) > 0:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_alignments = ["".join(template_hit_lines[n]) if n in template_hit_lines else None for n in Ms]

    return (a3m_lines, template_alignments) if use_templates else a3m_lines


def make_msas_colabfold_server(
    sequences: str | list[str],
    output_dir: PathLike,
    use_pairing: bool = False,
    use_templates: bool = False,
    user_agent: str = "",
    host_url: str = "https://api.colabfold.com",
    use_env: bool = True,
    use_filter: bool = True,
    pairing_strategy: Literal["greedy", "complete"] = "greedy",
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
) -> None:
    """Generate MSAs from protein sequences via the public ColabFold MSA server.

    Submits exactly ONE query: either an unpaired batch of independent sequences, or
    one paired/complex query. To generate both unpaired and paired MSAs for a full
    fine-tuning run, use :py:func:`make_msas_colabfold_server_batch`.

    Organizes the raw a3m results into AtomWorks' standard MSA layout via
    :py:func:`~atomworks.ml.preprocessing.msa.organizing.organize_msas` (unpaired) or
    :py:func:`~atomworks.ml.preprocessing.msa.organizing.organize_paired_msas`
    (paired).

    Msa results are discoverable through :py:func:`~atomworks.ml.preprocessing.msa.finding.find_paired_msas`

    Args:
        sequences: A single protein sequence string or list of protein sequences. For
            a paired query, the full unique-sequence set of one complex.
        output_dir: Base output directory
        use_pairing: Whether to submit a paired/complex query instead of independent
            per-sequence queries.
        use_templates: Whether to also fetch and organize raw template-alignment hits
            (``pdb70.m8`` rows) for each sequence, under ``<output_dir>/templates``.
            Ignored (forced off) when `use_pairing` is True, matching the ColabFold
            server's own constraint. To download the hit structures, see
            :py:mod:`~atomworks.ml.preprocessing.msa.template_structures`.
        user_agent: User agent string sent with the MSA server API calls.
        host_url: ColabFold MSA server URL.
        use_env: Whether to align against the environmental (metagenomic) database.
        use_filter: Whether to apply the ColabFold server's diversity filter.
        pairing_strategy: Pairing method, one of "greedy"/"complete" (only used when
            `use_pairing` is True).
        sharding_pattern: Directory sharding pattern (e.g. "/0:2/"), applied to both
            unpaired sequence hashes and, for paired output, complex_ids.
        output_extension: Output file extension (.a3m, .a3m.gz)
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    base_output_path = Path(output_dir)
    target_dir = base_output_path / ("paired" if use_pairing else "unpaired")

    effective_use_templates = use_templates and not use_pairing
    if use_templates and use_pairing:
        logger.warning("use_templates=True is ignored for paired queries (not supported by the ColabFold server).")

    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        result = query_colabfold_msa_server(
            sequences,
            prefix=scratch_path / "raw",
            user_agent=user_agent,
            host_url=host_url,
            use_env=use_env,
            use_filter=use_filter,
            use_pairing=use_pairing,
            use_templates=effective_use_templates,
            pairing_strategy=pairing_strategy,
        )
        a3m_lines, template_alignments = result if effective_use_templates else (result, None)

        # Write one raw a3m file per submitted sequence, mirroring the shape
        # make_msas_mmseqs's unpackdb step produces, so organize_msas/organize_paired_msas
        # can hash/shard them the same way regardless of which backend generated them.
        unpacked_dir = scratch_path / "unpacked"
        unpacked_dir.mkdir()
        for i, aln in enumerate(a3m_lines):
            (unpacked_dir / f"{i}.a3m").write_text(aln)

        org_config = MSAOrganizationConfig(
            input_extension=MSAFileExtension.A3M,
            output_extension=output_extension,
            sharding_pattern=sharding_pattern,
            copy_files=False,  # Move files instead of copying
        )
        logger.info("Organizing MSA files...")
        if use_pairing:
            organize_paired_msas(unpacked_dir, target_dir, org_config)
        else:
            organize_msas(unpacked_dir, target_dir, org_config)

        if effective_use_templates:
            sequence_to_template_alignment = dict(zip(sequences, template_alignments, strict=True))
            logger.info("Organizing template alignment files...")
            organize_template_alignments(
                sequence_to_template_alignment,
                base_output_path / "templates",
                sharding_pattern=sharding_pattern,
            )

    logger.info(f"MSA files saved to: {target_dir.absolute()}")


def make_msas_colabfold_server_batch(
    sequences: list[str],
    output_dir: PathLike,
    complexes: list[list[str]] | None = None,
    use_templates: bool = False,
    user_agent: str = "",
    host_url: str = "https://api.colabfold.com",
    use_env: bool = True,
    use_filter: bool = True,
    pairing_strategy: Literal["greedy", "complete"] = "greedy",
    sharding_pattern: str = "/0:2/",
    output_extension: str = MSAFileExtension.A3M_GZ.value,
) -> None:
    """Generate both unpaired and (optionally) paired MSAs for a full fine-tuning run,
    all under one `output_dir`.

    Msa results are written to the <output_dir> and has the following structure:
        - Unpaired (default): ``<output_dir>/unpaired/<shard>/<hash><ext>``,
          discoverable via :py:func:`~atomworks.ml.preprocessing.msa.finding.find_msas` /
          :py:func:`~atomworks.ml.transforms.msa._msa_loading_utils.get_msa_path`
          (point `dir` at ``<output_dir>/unpaired``).
        - Paired (`use_pairing=True`): submits a paired/complex query instead of
          independent per-sequence queries, and writes to
          ``<output_dir>/paired/<shard-of-complex_id>/<complex_id>/<chain_hash><ext>``
          where `complex_id` is computed based on the sequence hashes (see `get_complex_id`)
          and also sharded in the same way sequence hashes are.
        - Templates (`use_templates=True`): raw template-alignment hits for the
          unpaired sequences only (paired queries don't support templates), written
          to ``<output_dir>/templates/<shard>/<hash>.m8``, discoverable via
          :py:func:`~atomworks.ml.preprocessing.msa.finding.find_template_alignments`.

    Args:
        sequences: Unique protein sequences to submit as an independent (unpaired)
            batch. Every unique sequence appearing in `complexes` is automatically
            included as well, even if not passed here directly -- every chain needs
            its own unpaired MSA (and, if requested, template alignment) regardless
            of whether it also participates in pairing. Pass an empty list (with
            `complexes` also empty) to skip unpaired generation entirely.
        output_dir: Base output directory, shared by both unpaired and paired output
            -- see :py:func:`make_msas_colabfold_server`'s layout.
        complexes: Each entry is the full sequence set of one complex, submitted as a
            separate paired query. Entries with the same complex_id (same unique
            sequences, in any order or with repeats) are submitted once. None or empty
            to skip paired generation.
        use_templates: Whether to also fetch and organize raw template-alignment
            hits for `sequences` (the unpaired batch). See
            :py:func:`make_msas_colabfold_server`'s `use_templates` for details and
            limitations.
        user_agent: User agent string sent with the MSA server API calls.
        host_url: ColabFold MSA server URL.
        use_env: Whether to align against the environmental (metagenomic) database.
        use_filter: Whether to apply the ColabFold server's diversity filter.
        pairing_strategy: Pairing method for paired queries, one of "greedy"/"complete".
        sharding_pattern: Directory sharding pattern (e.g. "/0:2/").
        output_extension: Output file extension (.a3m, .a3m.gz)
    """
    common_kwargs = {
        "user_agent": user_agent,
        "host_url": host_url,
        "use_env": use_env,
        "use_filter": use_filter,
        "sharding_pattern": sharding_pattern,
        "output_extension": output_extension,
    }

    # Every chain needs its own unpaired MSA regardless of whether it also
    # participates in pairing -- union in complex members that weren't passed in
    # `sequences` directly, deduplicating while preserving first-seen order.
    unpaired_sequences = list(
        dict.fromkeys([*sequences, *(seq for complex_sequences in complexes or [] for seq in complex_sequences)])
    )

    if unpaired_sequences:
        make_msas_colabfold_server(unpaired_sequences, output_dir, use_templates=use_templates, **common_kwargs)

    # One paired query per complex_id -- a duplicate would waste a query and then hit
    # organize_paired_msas's existing-directory check.
    unique_complexes = {get_complex_id(complex_sequences): complex_sequences for complex_sequences in complexes or []}
    for complex_sequences in unique_complexes.values():
        make_msas_colabfold_server(
            complex_sequences,
            output_dir,
            use_pairing=True,
            pairing_strategy=pairing_strategy,
            **common_kwargs,
        )
