from collections.abc import Iterator, Sequence
from itertools import product
from string import ascii_uppercase


def create_chain_id_generator(unavailable_chain_ids: Sequence[str] = []) -> Iterator[str]:
    """
    Generate the next available chain ID that is not in the unavailable_chain_ids list.
    The chain IDs are generated in lexicographical order,
        i.e. A, B, C, ..., Z, AA, AB, ..., ZZ, AAA, etc.

    The first available chain ID will be returned, i.e. gaps in the unavailable_chain_ids
    list will be filled.

    Args:
        - unavailable_chain_ids (list[str]): List of already occupied chain IDs.

    Yields:
        - str: The next available chain ID.

    Example:
        >>> unavailable = ["A", "B", "C", "AA", "AB"]
        >>> next_id = create_chain_id_generator(unavailable)
        >>> print(next(next_id), next(next_id), next(next_id))
        D E F
    """
    unavailable_set = set(unavailable_chain_ids)

    def chain_id_generator() -> Iterator[str]:
        length = 1
        while True:
            for combo in product(ascii_uppercase, repeat=length):
                yield "".join(combo)
            length += 1

    for chain_id in chain_id_generator():
        if chain_id not in unavailable_set:
            yield chain_id
