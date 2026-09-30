"""Functions for generating and parsing example IDs that uniquely identify examples and their corresponding datasets."""

import re


def _parse_value(value: str) -> str | list:
    """Parse a value, evaluating list literals."""
    value = value.strip()
    if value.startswith("["):
        return eval(value)
    return value


def generate_example_id(
    dataset_names: list[str],
    pdb_id: str,
    assembly_id: str,
    query_pn_unit_iids: list,
    altloc_seed: int | None = None,
) -> str:
    """Generate a unique example ID from a DataFrame row.

    This unique ID is helpful for debugging and to track performance on specific examples.

    An example can be uniquely defined by, in order:
        (1) a composed list of dataset names (e.g., [pdb, pn_unit] to indicate the pn_unit dataset nested within the PDB dataset)
        (2) pdb_id (or any group-level identifier, if using a non-PDB dataset), within the dataset specified by (1)
        (3) assembly_id
        (4) query_pn_unit_iids
        (5) altloc_seed (optional) — distinguishes different alt-loc conformer samples of the same structure
    """
    # Format: {[dataset_names]}{pdb_id}{assembly_id}{query_pn_unit_iids}[{alt_loc_seed}]
    # Example for pn_unit dataset: {['pdb', 'pn_unit']}{6vyb}{1}{['A_1']}
    # Example for interface dataset: {['pdb', 'interfaces']}{6vyb}{1}{['A_1', 'B_1']}
    # Example for a distillation dataset: {['af2_distillation']}{6vyb}{1}{['A_1']}
    # Example with alt_loc_seed: {['pdb', 'interfaces']}{6vyb}{1}{['A_1', 'B_1']}{3}
    query_pn_unit_iids = [str(x) for x in query_pn_unit_iids]
    base = f"{{{dataset_names}}}{{{pdb_id}}}{{{assembly_id}}}{{{query_pn_unit_iids}}}"
    if altloc_seed is not None:
        base += f"{{{altloc_seed}}}"
    return base


def parse_example_id(
    example_id: str,
    keys: tuple[str, ...] = ("datasets", "pdb_id", "assembly_id", "query_pn_unit_iids"),
    *,
    permissive: bool = False,
) -> dict[str, str | list | None]:
    """Parse an example ID into its components.

    Args:
        example_id: The example ID string (e.g., ``{mgnify}{123}`` or ``{['pdb']}{6vyb}{1}{['A_1']}``).
        keys: Tuple of key names to map to extracted values, in order.
        permissive: If ``True``, return ``None`` for missing keys instead of raising.

    Returns:
        Dictionary mapping keys to parsed values. List literals are evaluated automatically.

    Raises:
        ValueError: If the number of extracted values doesn't match keys (when ``permissive=False``).

    Examples:
        >>> parse_example_id("{mgnify}{123}", keys=("datasets", "pdb_id"))
        {'datasets': 'mgnify', 'pdb_id': '123'}

        >>> parse_example_id("{['pdb']}{6vyb}{1}{['A_1']}")
        {'datasets': ['pdb'], 'pdb_id': '6vyb', 'assembly_id': '1', 'query_pn_unit_iids': ['A_1']}

        >>> parse_example_id("{mgnify}{123}", permissive=True)
        {'datasets': 'mgnify', 'pdb_id': '123', 'assembly_id': None, 'query_pn_unit_iids': None}

        >>> parse_example_id(
        ...     "{['pdb']}{6vyb}{1}{['A_1']}{3}",
        ...     keys=("datasets", "pdb_id", "assembly_id", "query_pn_unit_iids", "altloc_seed"),
        ... )
        {'datasets': ['pdb'], 'pdb_id': '6vyb', 'assembly_id': '1', 'query_pn_unit_iids': ['A_1'], 'alt_loc_seed': '3'}
    """
    matches = re.findall(r"\{(.*?)\}", example_id)

    if len(matches) != len(keys) and not permissive:
        raise ValueError(f"Expected {len(keys)} values for keys {keys}, got {len(matches)} from: {example_id}")

    result = {}
    for i, key in enumerate(keys):
        if i < len(matches):
            result[key] = _parse_value(matches[i])
        else:
            result[key] = None

    return result
