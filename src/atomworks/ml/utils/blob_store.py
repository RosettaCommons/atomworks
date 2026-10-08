"""Ranged-read store and memory-mapped index for per-record-zstd ``.bin`` shards on local disk or a remote store."""

import os

import pyarrow as pa
import pyarrow.feather as feather

from atomworks.ml.utils.huggingface import read_hf_bytes
from atomworks.ml.utils.io import (
    _decompress,
    build_feather_once,
    job_scoped_feather_path,
    read_parquet_with_metadata,
    read_s3_bytes,
)


class BlobStore:
    """Read one compressed record by ``(shard, offset, length)`` from a local directory or a remote store.

    Remote records use ranged GETs; ``endpoint_url`` applies only to S3. HF paths
    require an immutable revision. Only config is stored, so the store is picklable across workers.
    """

    def __init__(self, data_dir: str, *, endpoint_url: str | None = None):
        self.data_dir = str(data_dir).rstrip("/")
        self.endpoint_url = endpoint_url

    def get_bytes(self, shard: str, offset: int, length: int) -> bytes:
        """Fetch and decompress the record at ``[offset, offset+length)`` of ``shard``."""
        url = f"{self.data_dir}/{shard}"
        if url.startswith("hf://"):
            raw = read_hf_bytes(url, offset=offset, length=length)
        elif url.startswith("s3://"):
            raw = read_s3_bytes(url, offset=offset, length=length, endpoint_url=self.endpoint_url)
        elif "://" in url:
            raise ValueError("Blob stores support local paths, s3:// and revision-pinned hf://datasets/ URLs")
        else:
            with open(url, "rb") as f:
                f.seek(offset)
                raw = f.read(length)
        return _decompress(raw)


class BlobIndex:
    """Memory-mapped ``id_column`` -> ``(shard, offset, length)`` lookup for a blob store.

    Builds a node-local sorted feather once (under a lock), then mmaps it read-only and binary-searches it —
    so even a huge index is shared across DataLoader workers with ~no resident heap (``O(log N)`` lookups).
    """

    def __init__(
        self,
        index_path: str,
        *,
        id_column: str = "example_id",
        local_drive_mount: str | None = None,
        job_id_env_var: str = "SLURM_JOB_ID",
    ):
        self.index_path = str(index_path)
        self.id_column = id_column
        self.local_drive_mount = local_drive_mount or os.environ.get("LOCAL_DRIVE_MOUNT", "/tmp")
        self.job_id_env_var = job_id_env_var
        self.feather_path = self._build()
        self._open()

    def _build(self) -> str:
        """Build the sorted feather once (under a lock); return its path."""
        name = "blobidx_" + self.index_path.replace("://", "_").replace("/", "_").replace(":", "_")
        feather_path = job_scoped_feather_path(
            name,
            local_drive_mount=self.local_drive_mount,
            job_id_env_var=self.job_id_env_var,
            # `name` covers index_path only; build() also selects and sorts by id_column.
            content_key=self.id_column,
        )

        def build() -> pa.Table:
            df = read_parquet_with_metadata(self.index_path, columns=[self.id_column, "shard", "offset", "length"])
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            # Promote string -> large_string (int64 offsets): a huge id column exceeds Arrow's 2 GB
            # int32 string-offset limit, which would overflow in combine_chunks below.
            fields = [f.with_type(pa.large_string()) if f.type == pa.string() else f for f in tbl.schema]
            tbl = tbl.cast(pa.schema(fields))
            # Sort by id + collapse to one chunk so mmapped positional indexing is O(1) (binary search).
            return tbl.sort_by(self.id_column).combine_chunks()

        return build_feather_once(feather_path, build)

    def _open(self) -> None:
        self._table = feather.read_table(self.feather_path, memory_map=True)
        self._ids = self._table.column(self.id_column).combine_chunks()  # single mmapped StringArray
        self._shard = self._table.column("shard").combine_chunks()
        self._offset = self._table.column("offset").combine_chunks()
        self._length = self._table.column("length").combine_chunks()

    def __getstate__(self) -> dict:
        # Pickle the built feather path, not the mmapped arrays (re-mmapped per worker in __setstate__).
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open()

    def __len__(self) -> int:
        return self._table.num_rows

    def lookup(self, example_id: str) -> tuple[str, int, int]:
        """Return ``(shard, offset, length)`` for ``example_id``; raise ``KeyError`` if absent."""
        # Binary search over the build-time-sorted id column (Arrow byte-order sort == Python str compare
        # for UTF-8); a null id sorts first, so treat it as "less than".
        lo, hi, ids = 0, len(self._ids), self._ids
        while lo < hi:
            mid = (lo + hi) // 2
            mid_id = ids[mid].as_py()
            if mid_id is None or mid_id < example_id:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(ids) or ids[lo].as_py() != example_id:
            raise KeyError(f"{example_id!r} not found in blob index {self.index_path}")
        return self._shard[lo].as_py(), self._offset[lo].as_py(), self._length[lo].as_py()
