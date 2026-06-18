from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError


FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "RES200-local-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "RES200-local-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public files attached to a Figshare article.")
    parser.add_argument("--article-id", required=True, help="Figshare article ID, e.g. 29228990.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--metadata-only", action="store_true", help="Save metadata but do not download files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api_url = FIGSHARE_API.format(article_id=args.article_id)

    try:
        metadata = fetch_json(api_url)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Failed to fetch Figshare metadata: {exc}", file=sys.stderr)
        return 2

    (out / "figshare_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = metadata.get("files", [])
    if not files:
        print("No files found on Figshare article metadata.")
        return 0

    manifest = []
    for item in files:
        name = item.get("name") or f"figshare_file_{item.get('id', 'unknown')}"
        download_url = item.get("download_url")
        destination = out / name
        manifest.append(
            {
                "id": item.get("id"),
                "name": name,
                "size": item.get("size"),
                "download_url": download_url,
                "local_path": str(destination),
            }
        )
        if args.metadata_only:
            continue
        if not download_url:
            print(f"Skipping {name}: no download_url")
            continue
        print(f"Downloading {name}...")
        download_file(download_url, destination)

    (out / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved Figshare files/metadata to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

