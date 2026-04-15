#!/usr/bin/env python3
"""
Import seed data from pushshift/redarcs Reddit archives (.zst format).

Downloads zst archives from the-eye.eu and writes them directly to the data/
folder for consistency with the ongoing collection system. Does NOT merge
them into CSV files.

Usage (called by GitHub Actions, or manually):
    python import_seed_data.py [--data-dir DIR]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import zstandard

# ---------------------------------------------------------------------------
# Constants – must stay in sync with collect_reddit_data.py
# ---------------------------------------------------------------------------

SUBREDDITS = [
    "autism", "aspergers", "aspergirls", "AutisticAdults",
    "ADHD", "ADHDmemes", "adhdwomen", "adhd_anxiety",
]

# Base URL for downloading archives from the-eye.eu
BASE_URL = "https://the-eye.eu/redarcs/files"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_author(username: str) -> str:
    """Return first 16 hex chars of SHA-256(username), empty for deleted."""
    if not username or username in ("[deleted]", "[removed]"):
        return ""
    return hashlib.sha256(username.encode()).hexdigest()[:16]


def _utc_date(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _iter_ndjson_zst(zst_path: Path):
    """Yield parsed JSON objects from a zstd-compressed NDJSON file."""
    dctx = zstandard.ZstdDecompressor()
    buf = b""
    with open(zst_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        while True:
            chunk = reader.read(131_072)
            if not chunk:
                break
            buf += chunk
            lines = buf.split(b"\n")
            buf = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
    if buf.strip():
        try:
            yield json.loads(buf)
        except json.JSONDecodeError:
            pass


# ---------------------------------------------------------------------------
# Download and processing helpers
# ---------------------------------------------------------------------------

def download_archive(subreddit: str, kind: str, dest_dir: Path) -> Path:
    """
    Download a single archive from the-eye.eu.

    Parameters
    ----------
    subreddit : subreddit name
    kind : 'submissions' or 'comments'
    dest_dir : destination directory for the file

    Returns the path to the downloaded file.
    """
    filename = f"{subreddit}_{kind}.zst"
    dest_path = dest_dir / filename
    url = f"{BASE_URL}/{filename}"

    print(f"  Downloading {filename} from the-eye.eu...")
    # Use curl with --insecure flag due to invalid SSL certificate on the-eye.eu
    cmd = [
        "curl", "-L", "--insecure", "-f", "-o", str(dest_path),
        "-A", "python:reddit_audhd:v1.0",
        url
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"    ERROR: Failed to download {filename}")
        print(f"    {result.stderr.decode()}")
        if dest_path.exists():
            dest_path.unlink()
        return None

    print(f"    Downloaded {dest_path.stat().st_size:,} bytes")
    return dest_path


def normalize_and_write_zst(input_zst: Path, output_zst: Path, kind: str) -> int:
    """
    Read archive, normalize fields to match collect_reddit_data.py format,
    and write to output zst file.

    Parameters
    ----------
    input_zst : path to input zst file
    output_zst : path to output zst file
    kind : 'submissions' or 'comments'

    Returns the number of records written.
    """
    print(f"  Normalizing {input_zst.name}...")

    items = {}
    for obj in _iter_ndjson_zst(input_zst):
        obj_id = obj.get("id", "")
        if not obj_id:
            continue

        if kind == "submissions":
            ts = obj.get("created_utc", 0)
            items[obj_id] = {
                "id": obj_id,
                "subreddit": obj.get("subreddit", ""),
                "title": obj.get("title", ""),
                "selftext": obj.get("selftext", ""),
                "author_hash": _hash_author(obj.get("author", "")),
                "score": obj.get("score", 0),
                "upvote_ratio": obj.get("upvote_ratio", ""),
                "num_comments": obj.get("num_comments", 0),
                "created_utc": ts,
                "created_date": _utc_date(ts),
                "url": obj.get("url", ""),
                "is_self": obj.get("is_self", False),
                "permalink": obj.get("permalink", ""),
            }
        elif kind == "comments":
            ts = obj.get("created_utc", 0)
            items[obj_id] = {
                "id": obj_id,
                "subreddit": obj.get("subreddit", ""),
                "body": obj.get("body", ""),
                "author_hash": _hash_author(obj.get("author", "")),
                "score": obj.get("score", 0),
                "created_utc": ts,
                "created_date": _utc_date(ts),
                "parent_id": obj.get("parent_id", ""),
                "link_id": obj.get("link_id", ""),
            }

    # Write to output file
    cctx = zstandard.ZstdCompressor()
    with open(output_zst, 'wb') as fh:
        with cctx.stream_writer(fh) as writer:
            for item in items.values():
                line = json.dumps(item) + '\n'
                writer.write(line.encode('utf-8'))

    print(f"    Wrote {len(items):,} records to {output_zst.name}")
    return len(items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import seed datasets from the-eye.eu Reddit archives."
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory where zst archives will be stored (default: data/)",
    )
    parser.add_argument(
        "--temp-dir", default="/tmp/seed_download",
        help="Temporary directory for downloads (default: /tmp/seed_download)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Download and import even when archives already exist.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    temp_dir = Path(args.temp_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Seed data import from the-eye.eu")
    print(f"Data directory: {data_dir}")
    print(f"Temp directory: {temp_dir}")
    print()

    # Check if we should skip import
    if not args.force:
        # Check if we already have substantial data
        existing_count = 0
        for subreddit in SUBREDDITS:
            sub_file = data_dir / f"{subreddit}_submissions.zst"
            if sub_file.exists():
                existing_count += 1

        if existing_count >= len(SUBREDDITS):
            print(
                f"All {len(SUBREDDITS)} subreddit archives already exist in {data_dir}. "
                "Skipping download (use --force to override)."
            )
            return

    # Download and process each subreddit
    total_subs = 0
    total_coms = 0

    for subreddit in SUBREDDITS:
        print(f"\n{'='*60}")
        print(f"Processing: r/{subreddit}")
        print('='*60)

        for kind in ['submissions', 'comments']:
            output_file = data_dir / f"{subreddit}_{kind}.zst"

            # Skip if already exists and not forced
            if output_file.exists() and not args.force:
                print(f"  {output_file.name} already exists, skipping")
                continue

            # Download to temp directory
            temp_file = download_archive(subreddit, kind, temp_dir)
            if not temp_file or not temp_file.exists():
                print(f"  Failed to download {subreddit}_{kind}.zst")
                continue

            # Normalize and write to data directory
            count = normalize_and_write_zst(temp_file, output_file, kind)

            if kind == 'submissions':
                total_subs += count
            else:
                total_coms += count

            # Clean up temp file
            temp_file.unlink()

    print(
        f"\n{'='*60}\n"
        f"Seed import complete\n"
        f"{'='*60}\n"
        f"Total submissions: {total_subs:,}\n"
        f"Total comments: {total_coms:,}\n"
        f"Files written to: {data_dir}/\n"
    )


if __name__ == "__main__":
    main()
