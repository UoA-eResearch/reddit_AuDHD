#!/usr/bin/env python3
"""
Import seed data from pushshift/redarcs Reddit archives (.zst format).

Downloads the redarcs torrent via aria2c, selects only the subreddits we
track, and merges the resulting data into our CSV files using the same
field layout and author-hashing convention as collect_reddit_data.py.

Usage (called by GitHub Actions, or manually):
    python import_seed_data.py [--seed-dir DIR] [--force]
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import zstandard
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants – must stay in sync with collect_reddit_data.py
# ---------------------------------------------------------------------------

SUBREDDITS = [
    "autism", "aspergers", "aspergirls", "AutisticAdults",
    "ADHD", "ADHDmemes", "adhdwomen", "adhd_anxiety",
]

MAGNET_URI = (
    "magnet:?xt=urn:btih:3e3f64dee22dc304cdd2546254ca1f8e8ae542b4"
    "&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php"
    "&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969"
    "&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
)

SUBMISSION_COLS = [
    "id", "subreddit", "title", "selftext", "author_hash", "score",
    "upvote_ratio", "num_comments", "created_utc", "created_date",
    "url", "is_self", "permalink",
]

COMMENT_COLS = [
    "id", "subreddit", "body", "author_hash", "score",
    "created_utc", "created_date", "parent_id", "link_id",
]


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


def _load_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def import_submissions(zst_path: Path, out_csv: Path) -> int:
    existing_ids = _load_existing_ids(out_csv)
    new_rows = []
    for post in _iter_ndjson_zst(zst_path):
        pid = post.get("id", "")
        if not pid or pid in existing_ids:
            continue
        ts = post.get("created_utc", 0)
        new_rows.append({
            "id": pid,
            "subreddit": post.get("subreddit", ""),
            "title": post.get("title", ""),
            "selftext": post.get("selftext", ""),
            "author_hash": _hash_author(post.get("author", "")),
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", ""),
            "num_comments": post.get("num_comments", 0),
            "created_utc": ts,
            "created_date": _utc_date(ts),
            "url": post.get("url", ""),
            "is_self": post.get("is_self", False),
            "permalink": post.get("permalink", ""),
        })
        existing_ids.add(pid)

    if new_rows:
        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUBMISSION_COLS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    return len(new_rows)


def import_comments(zst_path: Path, out_csv: Path) -> int:
    existing_ids = _load_existing_ids(out_csv)
    new_rows = []
    for comment in _iter_ndjson_zst(zst_path):
        cid = comment.get("id", "")
        if not cid or cid in existing_ids:
            continue
        ts = comment.get("created_utc", 0)
        new_rows.append({
            "id": cid,
            "subreddit": comment.get("subreddit", ""),
            "body": comment.get("body", ""),
            "author_hash": _hash_author(comment.get("author", "")),
            "score": comment.get("score", 0),
            "created_utc": ts,
            "created_date": _utc_date(ts),
            "parent_id": comment.get("parent_id", ""),
            "link_id": comment.get("link_id", ""),
        })
        existing_ids.add(cid)

    if new_rows:
        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COMMENT_COLS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    return len(new_rows)


# ---------------------------------------------------------------------------
# Torrent helpers
# ---------------------------------------------------------------------------

def download_torrent(magnet_uri: str, dest_dir: Path) -> bool:
    """Download files from the torrent using aria2c with a magnet link."""
    cmd = [
        "aria2c",
        "--seed-time=0",
        "--file-allocation=none",
        f"--dir={dest_dir}",
        magnet_uri,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import seed datasets from the redarcs torrent."
    )
    parser.add_argument(
        "--seed-dir", default="/tmp/seed",
        help="Directory where torrent files are downloaded.",
    )
    parser.add_argument(
        "--submissions-csv", default="reddit_submissions.csv",
    )
    parser.add_argument(
        "--comments-csv", default="reddit_comments.csv",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Import even when the CSV files already have substantial data.",
    )
    args = parser.parse_args()

    subs_csv = Path(args.submissions_csv)
    coms_csv = Path(args.comments_csv)
    seed_dir = Path(args.seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Skip if we already have substantial data (avoids re-seeding every run).
    if not args.force and subs_csv.exists():
        with open(subs_csv) as f:
            lines = sum(1 for _ in f) - 1  # exclude header
        if lines > 10_000:
            print(
                f"Already have {lines:,} submissions; skipping seed import "
                "(use --force to override)."
            )
            return

    # Download via aria2c using magnet link.
    print(f"Starting torrent download via aria2c (magnet link) ...")
    if not download_torrent(MAGNET_URI, seed_dir):
        print("aria2c download failed.", file=sys.stderr)
        sys.exit(1)

    # Process downloaded .zst archives.
    total_subs = 0
    total_coms = 0
    sub_lower = {s.lower() for s in SUBREDDITS}

    for zst_file in sorted(seed_dir.rglob("*.zst")):
        stem = zst_file.stem  # e.g. "autism_submissions" or "autism_comments"
        if "_" not in stem:
            continue
        sub_name, kind = stem.rsplit("_", 1)
        if sub_name.lower() not in sub_lower:
            continue

        if kind == "submissions":
            print(f"Importing submissions from {zst_file.name} ...")
            n = import_submissions(zst_file, subs_csv)
            total_subs += n
            print(f"  +{n:,} submissions")
        elif kind == "comments":
            print(f"Importing comments from {zst_file.name} ...")
            n = import_comments(zst_file, coms_csv)
            total_coms += n
            print(f"  +{n:,} comments")

    print(
        f"\nSeed import complete: "
        f"+{total_subs:,} submissions, +{total_coms:,} comments"
    )


if __name__ == "__main__":
    main()
