"""Content-addressed packed MSA store: ``hash_sequence(seq, MSA_KEY_LEN)`` -> ``(shard, offset, length)`` -> bytes.

Layout (local path or ``s3://`` URL): ``<store>/index.parquet`` (columns hash, shard, offset, length) +
``<store>/data/*.bin`` (per-record-zstd a3m). Keyed via :func:`~atomworks.ml.utils.misc.hash_sequence`. Reads go through :class:`~atomworks.ml.utils.blob_store.BlobStore`.
"""

from collections.abc import Sequence

from atomworks.ml.utils.blob_store import BlobStore
from atomworks.ml.utils.io import read_parquet_with_metadata
from atomworks.ml.utils.misc import hash_sequence

# Canonical MSA content-address key: 72-bit sha256 prefix (~1e-6 collision odds at 100M).
MSA_KEY_LEN = 18
# Legacy 11-char key; probed as a fallback so stores built before MSA_KEY_LEN still resolve.
LEGACY_MSA_KEY_LEN = 11


class PackedMsaStore:
    """Look up an MSA's bytes by its ``hash_sequence`` digest.

    Args:
      store_url: Store root (local path or ``s3://`` URL) holding ``index.parquet`` + ``data/*.bin``.
      endpoint_url: Endpoint URL for ``s3://`` stores (defaults to the ambient AWS config).
      key_lens: Digest lengths to probe, in order — canonical ``MSA_KEY_LEN`` first, then legacy fallbacks —
        so a store built with an older key length still resolves.
    """

    def __init__(
        self,
        store_url: str,
        *,
        endpoint_url: str | None = None,
        key_lens: Sequence[int] = (MSA_KEY_LEN, LEGACY_MSA_KEY_LEN),
    ):
        store_url = str(store_url).rstrip("/")
        self._index_url = f"{store_url}/index.parquet"
        self._blob = BlobStore(f"{store_url}/data", endpoint_url=endpoint_url)
        self._key_lens = tuple(key_lens)
        self._index: dict[str, tuple[str, int, int]] | None = None  # lazy per-worker

    def _get_index(self) -> dict[str, tuple[str, int, int]]:
        if self._index is None:
            df = read_parquet_with_metadata(self._index_url)
            self._index = {
                h: (s, int(o), int(length))
                for h, s, o, length in zip(df["hash"], df["shard"], df["offset"], df["length"], strict=True)
            }
        return self._index

    def get_bytes_for_seq(self, seq: str) -> bytes | None:
        """Return the (decompressed) MSA bytes for ``seq``, or ``None`` if it is not in the store.

        Probes each length in ``key_lens`` (canonical first, legacy fallback next) so older stores still resolve.
        """
        index = self._get_index()
        for length in self._key_lens:
            loc = index.get(hash_sequence(seq, length=length))
            if loc is not None:
                return self._blob.get_bytes(*loc)
        return None
