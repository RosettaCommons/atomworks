import pytest
import zstandard as zstd

from atomworks.io.utils.compression import open_compressed, transfer_with_compression
from tests.io.conftest import TEST_DATA_IO


@pytest.mark.parametrize("suffix", ["", ".gz", ".gzip", ".zst"])
@pytest.mark.parametrize("move", [False, True])
def test_transfer_codecs(tmp_path, compressed_example, suffix, move):
    original = TEST_DATA_IO / "1a8o_modified.cif"
    encoded = compressed_example(original)
    source = tmp_path / ("source." + encoded.name)
    source.write_bytes(encoded.read_bytes())
    destination = tmp_path / ("output.cif" + suffix)
    transfer_with_compression(source, destination, move=move)
    with open_compressed(destination, "rb") as stream:
        assert stream.read() == original.read_bytes()
    assert source.exists() != move


@pytest.mark.parametrize("damage", [None, "truncated", "corrupt", "empty"])
def test_zstd_transfer_completion(tmp_path, damage):
    data = (TEST_DATA_IO / "1a8o_modified.cif").read_bytes()
    compressor = zstd.ZstdCompressor(write_checksum=True, write_content_size=False)
    encoded = compressor.compress(data) + compressor.compress(data)
    encoded = {"truncated": encoded[:-1], "corrupt": encoded[:-1] + bytes([encoded[-1] ^ 1]), "empty": b""}.get(
        damage, encoded
    )
    source, destination = tmp_path / "input.cif.zst", tmp_path / "output.cif"
    source.write_bytes(encoded)
    destination.write_bytes(b"keep")
    if damage is None:
        with open_compressed(source, "rb") as stream:
            assert b"".join(iter(lambda: stream.read(7), b"")) == data * 2
        transfer_with_compression(source, destination, move=True)
        assert destination.read_bytes() == data * 2
        assert not source.exists()
    else:
        with pytest.raises((EOFError, zstd.ZstdError)):
            transfer_with_compression(source, destination, move=True)
        assert source.read_bytes() == encoded
        assert destination.read_bytes() == b"keep"
    assert set(tmp_path.iterdir()) == ({destination} if damage is None else {source, destination})
