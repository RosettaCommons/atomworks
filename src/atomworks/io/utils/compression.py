"""Compression utilities for file operations."""

__all__ = [
    "compress_file",
    "is_compressed_file",
    "maybe_decompress_file",
    "open_compressed",
    "transfer_with_compression",
]

import errno
import gzip
import io
import os
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

import zstandard as zstd

Compression = Literal["gzip", "zstd"] | None


def _compression_format(path: PathLike | str) -> Compression:
    suffix = Path(path).suffix.lower()
    if suffix in (".gz", ".gzip"):
        return "gzip"
    return "zstd" if suffix == ".zst" else None


def is_compressed_file(path: PathLike | str) -> bool:
    """Check if a file is compressed based on its extension."""
    return _compression_format(path) is not None


def _open_stream(
    path: PathLike | str, mode: str, compression: Compression, encoding: str | None = None
) -> BinaryIO | TextIO:
    kwargs = {"encoding": encoding} if "t" in mode else {}
    if compression == "gzip":
        return gzip.open(path, mode, **kwargs)
    if compression == "zstd":
        if "r" in mode:
            reader = io.BufferedReader(zstd.open(path, "rb"))
            return io.TextIOWrapper(reader, encoding=encoding) if "t" in mode else reader
        writer = io.BufferedWriter(zstd.open(path, "wb", cctx=zstd.ZstdCompressor(level=3)))
        return io.TextIOWrapper(writer, encoding=encoding) if "t" in mode else writer
    return open(path, mode, **kwargs)


def open_compressed(path: PathLike | str, mode: str = "rt", *, encoding: str | None = None) -> BinaryIO | TextIO:
    """Open a file, selecting compression by suffix.

    Args:
        path: File path. ``.gz``/``.gzip`` use gzip; ``.zst`` uses Zstandard.
        mode: ``rt``, ``rb``, ``wt``, or ``wb``. Bare ``r``/``w`` mean text mode.
        encoding: Text encoding; defaults to the platform encoding.

    Returns:
        A stream that the caller must close, typically using a ``with`` block.
        Zstandard writes use level 3; gzip writes use its default level 9.
        Zstandard reads support concatenated frames and frames without a content size.
    """
    if mode in ("r", "w"):
        mode += "t"
    if mode not in ("rt", "rb", "wt", "wb"):
        raise ValueError(f"Unsupported mode {mode!r}; use rt, rb, wt, or wb")
    if "b" in mode and encoding is not None:
        raise ValueError("encoding is only supported in text mode")
    return _open_stream(path, mode, _compression_format(path), encoding)


def _copy_zstd(source: BinaryIO, destination: BinaryIO) -> None:
    """Decompress concatenated frames, requiring a complete final frame."""
    decoder = zstd.ZstdDecompressor().decompressobj()
    while chunk := source.read(zstd.DECOMPRESSION_RECOMMENDED_INPUT_SIZE):
        while chunk:
            if decoder.eof:
                decoder = zstd.ZstdDecompressor().decompressobj()
            destination.write(decoder.decompress(chunk))
            chunk = decoder.unused_data
    if not decoder.eof:
        raise EOFError("Zstandard file ended before the end of a complete frame")


def _transfer(
    input_path: Path,
    output_path: Path,
    input_compression: Compression,
    output_compression: Compression,
    remove_original: bool,
) -> Path:
    """Stream into a temporary file, replacing the destination only after success."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve() or (output_path.exists() and input_path.samefile(output_path)):
        raise shutil.SameFileError(f"Input and output refer to the same file: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_compression == output_compression and remove_original:
        try:
            os.replace(input_path, output_path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
        else:
            return output_path
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if input_compression == output_compression:
            shutil.copy2(input_path, temporary_path)
        else:
            with (
                _open_stream(input_path, "rb", None if input_compression == "zstd" else input_compression) as source,
                _open_stream(temporary_path, "wb", output_compression) as destination,
            ):
                if input_compression == "zstd":
                    _copy_zstd(source, destination)
                else:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            shutil.copymode(input_path, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if remove_original:
        input_path.unlink()
    return output_path


def compress_file(
    input_file: PathLike | str, output_file: PathLike | str | None = None, remove_original: bool = False
) -> Path:
    """Compress raw file bytes, selecting compression by the output suffix.

    Args:
        input_file: File to compress. Its contents are treated as uncompressed bytes.
        output_file: Destination. Defaults to the input path with ``.gz`` appended.
        remove_original: Remove the source only after a successful transfer.

    Returns:
        The compressed file path. Input and output must be distinct files.
    """
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else input_path.with_suffix(input_path.suffix + ".gz")
    return _transfer(input_path, output_path, None, _compression_format(output_path) or "gzip", remove_original)


def maybe_decompress_file(
    input_file: PathLike | str, output_file: PathLike | str | None = None, remove_original: bool = False
) -> Path:
    """Decompress a file only if compressed, otherwise return its path unchanged.

    Args:
        input_file: File whose suffix determines compression.
        output_file: Destination. Defaults to removing the final compression suffix.
        remove_original: Remove the source only after a successful transfer.

    Returns:
        The uncompressed path. For compressed inputs, the destination must be distinct.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    compression = _compression_format(input_path)
    if compression is None:
        return input_path
    output_path = Path(output_file) if output_file is not None else input_path.with_suffix("")
    return _transfer(input_path, output_path, compression, None, remove_original)


def transfer_with_compression(input_file: PathLike | str, output_file: PathLike | str, move: bool = False) -> Path:
    """Copy or move a file with automatic compression/decompression based on file extensions.

    Equal codecs are copied byte-for-byte; different codecs are streamed through
    decompression/compression. Zstandard writes use level 3. The source is removed
    only after success when ``move=True``. Input and output must be distinct files.
    Same-codec moves use a rename when source and destination share a filesystem.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    return _transfer(input_path, output_path, _compression_format(input_path), _compression_format(output_path), move)
