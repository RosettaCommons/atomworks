"""Point archived PyData theme pages at the deployed site's version manifest."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

SWITCHER_URL = re.compile(r"(DOCUMENTATION_OPTIONS\.theme_switcher_json_url\s*=\s*)([\"'])(.*?)(\2)")


def repair_switcher_urls(pages: Path, site_url: str) -> int:
    """Update only switcher URL assignments, preserving archived content and version labels."""
    parsed = urlsplit(site_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Expected an HTTP(S) documentation site URL, got {site_url!r}")
    manifest_url = site_url.rstrip("/") + "/latest/_static/switcher.json"
    manifest = pages / "latest" / "_static" / "switcher.json"
    entries = json.loads(manifest.read_text())
    if not entries:
        raise ValueError(f"Empty version manifest: {manifest}")
    changed = 0
    for page in pages.rglob("*.html"):
        original = page.read_text()
        updated = SWITCHER_URL.sub(lambda match: match[1] + json.dumps(manifest_url), original)
        if updated != original:
            page.write_text(updated)
            changed += 1
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", type=Path)
    parser.add_argument("site_url")
    args = parser.parse_args()
    print(f"Updated version switcher URLs in {repair_switcher_urls(args.pages, args.site_url)} pages")
