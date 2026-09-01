#!/usr/bin/env python3
"""Build-time weight downloader with bounded retries and atomic replacement.

The source argument may be one URL or a ``|``-separated ordered list of URLs.
The latter keeps large GitHub Release assets reproducible on upload paths that
require multipart transfer.  Parts are joined byte-for-byte before the
Dockerfile checks the canonical archive digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path


def _download_one(url: str, destination: Path) -> None:
    for attempt in range(1, 7):
        try:
            urllib.request.urlretrieve(url, destination)
            return
        except Exception as error:
            destination.unlink(missing_ok=True)
            if attempt == 6:
                raise
            print(f"download attempt {attempt}/6 failed: {error!r}", file=sys.stderr)
            time.sleep(5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weight_spec(source_spec: str, output: Path) -> None:
    urls = [item for item in source_spec.split("|") if item]
    if not urls:
        raise ValueError("weight source must contain at least one URL")
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".partial")
    part = output.with_suffix(output.suffix + ".part.partial")
    try:
        with temporary.open("wb") as merged:
            for index, url in enumerate(urls):
                _download_one(url, part)
                with part.open("rb") as source:
                    shutil.copyfileobj(source, merged, length=16 * 1024 * 1024)
                print(f"downloaded part {index + 1}/{len(urls)} {url} bytes={part.stat().st_size}")
                part.unlink()
        os.replace(temporary, output)
    except BaseException:
        part.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    print(f"assembled {len(urls)} source(s) -> {output} bytes={output.stat().st_size}")


def download_manifest_archive(
    manifest_path: Path,
    archive_key: str,
    release_base_url: str,
    output: Path,
) -> None:
    """Download, verify, and join one archive declared by the release manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    archive = manifest["archives"][archive_key]
    parts = archive.get("multipart_join_order", [])
    if not parts:
        raise ValueError(f"archive {archive_key!r} has no multipart records")
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".partial")
    part_path = output.with_suffix(output.suffix + ".part.partial")
    try:
        with temporary.open("wb") as merged:
            for index, record in enumerate(parts):
                url = f"{release_base_url.rstrip('/')}/{record['name']}"
                _download_one(url, part_path)
                observed_bytes = part_path.stat().st_size
                observed_sha256 = _sha256(part_path)
                if observed_bytes != int(record["bytes"]):
                    raise ValueError(
                        f"part size mismatch for {record['name']}: "
                        f"{observed_bytes} != {record['bytes']}"
                    )
                if observed_sha256 != record["sha256"]:
                    raise ValueError(f"part sha256 mismatch for {record['name']}")
                with part_path.open("rb") as source:
                    shutil.copyfileobj(source, merged, length=16 * 1024 * 1024)
                part_path.unlink()
                print(
                    f"verified part {index + 1}/{len(parts)} "
                    f"{record['name']} bytes={observed_bytes}"
                )
        if temporary.stat().st_size != int(archive["bytes"]):
            raise ValueError(f"archive size mismatch for {archive_key!r}")
        if _sha256(temporary) != archive["sha256"]:
            raise ValueError(f"archive sha256 mismatch for {archive_key!r}")
        os.replace(temporary, output)
    except BaseException:
        part_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    print(f"assembled verified archive {archive_key} -> {output} bytes={output.stat().st_size}")


def main() -> None:
    if len(sys.argv) == 6 and sys.argv[1] == "--manifest":
        _, _, manifest, archive_key, release_base_url, output = sys.argv
        download_manifest_archive(
            Path(manifest), archive_key, release_base_url, Path(output)
        )
        return
    if len(sys.argv) != 3:
        raise SystemExit(
            f"usage: {sys.argv[0]} URL_OR_PIPE_SEPARATED_URLS OUTPUT\n"
            f"   or: {sys.argv[0]} --manifest MANIFEST ARCHIVE_KEY RELEASE_BASE_URL OUTPUT"
        )
    source_spec, output_text = sys.argv[1:]
    download_weight_spec(source_spec, Path(output_text))


if __name__ == "__main__":
    main()
