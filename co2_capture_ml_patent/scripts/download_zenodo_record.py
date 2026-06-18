from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError


ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "RES200-local-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "RES200-local-research/1.0"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=180) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected public files from a Zenodo record.")
    parser.add_argument("--record-id", required=True, help="Zenodo record ID, e.g. 3251643.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional filenames to download. Default: all files.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        metadata = fetch_json(ZENODO_API.format(record_id=args.record_id))
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Failed to fetch Zenodo metadata: {exc}", file=sys.stderr)
        return 2

    (out / "zenodo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    requested = set(args.files or [])
    manifest = []
    for item in metadata.get("files", []):
        key = item.get("key")
        if requested and key not in requested:
            continue
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        destination = out / str(key)
        manifest.append(
            {
                "key": key,
                "size": item.get("size"),
                "checksum": item.get("checksum"),
                "url": url,
                "local_path": str(destination),
            }
        )
        if args.metadata_only:
            continue
        if not url:
            print(f"Skipping {key}: no file URL")
            continue
        print(f"Downloading {key}...")
        download_file(url, destination)

    (out / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Zenodo metadata/files to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

