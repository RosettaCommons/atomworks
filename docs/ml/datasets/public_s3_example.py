"""Download a small CIF subset through anonymous S3 byte-range reads.

Install AtomWorks with its ml and s3 extras, then run:
    python public_s3_example.py --count 2 --output-dir tiny-tcr
"""

import argparse
import json
from pathlib import Path

from atomworks.ml.utils.blob_store import BlobIndex, BlobStore
from atomworks.ml.utils.io import S3ReadConfig, read_parquet_with_metadata, to_parquet_with_metadata

PREFIX = "s3://rfd4-proteina-public-e04a/train_datasets/streaming-latest/synthetic/2026_09_09_tcr_af2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("tiny-tcr"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = S3ReadConfig(endpoint_url="https://cwobject.com", anonymous=True)
    examples = read_parquet_with_metadata(f"{PREFIX}/examples.parquet", s3_config=config).head(args.count).copy()
    index = BlobIndex(f"{PREFIX}/blob/index.parquet", id_column="path", s3_config=config)
    store = BlobStore(f"{PREFIX}/blob/data", s3_config=config)
    paths = {}
    for record_id in examples["path"].unique():
        name = f"record-{len(paths):04d}.cif"
        (args.output_dir / name).write_bytes(store.get_bytes(*index.lookup(record_id)))
        paths[record_id] = name
    examples["source_record_id"] = examples["path"]
    examples["path"] = examples["path"].map(paths)
    examples.attrs = {"source": PREFIX}
    to_parquet_with_metadata(examples, args.output_dir / "examples.parquet")
    print(json.dumps({"examples": len(examples), "structures": len(paths), "output": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
