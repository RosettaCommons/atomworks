# Security constraints for new environments

Use these constraints when creating or updating an environment from this checkout:

```bash
uv pip install -c constraints/security.txt ".[ml,ase,dev,docs]"
# Or select only the extras you need; for core I/O:
uv pip install -c constraints/security.txt .
```

Pip accepts the same `-c constraints/security.txt` option. Constraints restrict packages only when your selected dependencies require them; they do not install every listed package or make optional features core dependencies.

The library retains `torch>=2.2.0` for compatibility with existing model environments. That broad range includes versions with known advisories. This optional constraints file instead selects Torch 2.14.1 or newer for new ML environments. Evaluate checkpoint, CUDA and downstream compatibility before updating an existing model environment. Installing without this file does not apply these additional restrictions.

The package manifest enforces `tqdm>=4.66.3` and the dev extra enforces `pytest>=9.0.3`. The existing exact Biotite 1.6.0 pin and PyArrow minimum are preserved. Transitive security floors live here instead of adding unrelated core dependencies.

## Verification

Checked October 1, 2026 with uv 0.12.19 and pip-audit 2.10.1. Linux x86_64 resolutions for Python 3.11 and 3.12 with all declared extras and these constraints had no reported vulnerabilities. This checks dependency resolution and published advisory data, not runtime or GPU compatibility. It is not an exhaustive guarantee for every Python version or platform.

To repeat the check from the repository root:

```bash
uv pip compile pyproject.toml --all-extras --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --resolution lowest-direct \
  --constraints constraints/security.txt --no-annotate --output-file /tmp/atomworks-security.txt
uv tool run --from pip-audit pip-audit --no-deps --disable-pip \
  -r /tmp/atomworks-security.txt
```

Repeat with `--python-version 3.11`. Refresh the constraints as advisories change; an old environment is not updated by checking out a new constraints file.

Representative advisories motivating the floors include [urllib3 streaming](https://github.com/advisories/GHSA-vxq7-64xx-v4gw), [Pillow image transforms](https://github.com/advisories/GHSA-9hw9-ch79-4vh6), [cryptography PKCS#7](https://github.com/advisories/GHSA-g6cj-pr64-35w5), [Tornado response limits](https://github.com/advisories/GHSA-chx6-46f5-w4vp), and [tqdm CLI arguments](https://github.com/advisories/GHSA-g7vv-2v7x-gj9p). The audit covers the complete resolved inventory, including Torch and pytest.
