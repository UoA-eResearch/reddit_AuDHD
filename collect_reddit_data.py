#!/usr/bin/env python3
"""
Script to collect Reddit posts and comments about Autism and ADHD using the
official Reddit JSON API (no authentication required).

The Reddit API exposes public JSON endpoints at:
  https://www.reddit.com/r/{subreddit}/{listing}.json

We paginate through the 'new' listing (most recent → oldest) and also pull
all-time 'top' posts to capture as much historical content as possible.
For each submission we fetch its comments via:
  https://www.reddit.com/r/{subreddit}/comments/{id}.json

Tor routing & exit-node rotation
---------------------------------
When a Tor SOCKS5 daemon is running on 127.0.0.1:9050 (or the TOR_PROXY env
var is set), all requests are routed through Tor so that Reddit's datacenter-IP
blocks are bypassed.

If Reddit returns 429 (rate-limited) or 403 (blocked) while using Tor, the
script automatically restarts the Tor daemon to obtain a fresh exit node and
retries the request.  This is transparent to callers.

Run with Tor (explicit proxy):
    TOR_PROXY=socks5h://127.0.0.1:9050 python3 collect_reddit_data.py
Run with torsocks wrapper:
    torsocks python3 collect_reddit_data.py
Seed-only run (1 page per subreddit, no comments – quick proof-of-concept):
    python3 collect_reddit_data.py --seed
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import zstandard
from tqdm import tqdm

# Subreddits focused on Autism and ADHD
AUTISM_SUBREDDITS = ['autism', 'aspergers', 'aspergirls', 'AutisticAdults']
ADHD_SUBREDDITS = ['ADHD', 'ADHDmemes', 'adhdwomen', 'adhd_anxiety']
ALL_SUBREDDITS = AUTISM_SUBREDDITS + ADHD_SUBREDDITS

# Reddit requires a descriptive User-Agent string for API access
HEADERS = {
    'User-Agent': 'python:reddit_audhd_research:v2.0 (academic research on neurodivergent communities)'
}

# Reddit caps listings at 100 items per request and ~1000 items total per listing
LIMIT = 100
# Delay between requests to stay well within Reddit's rate limit (~30 req/min)
REQUEST_DELAY = 2.0
# Maximum comments to fetch per post (top-level only)
MAX_COMMENTS_PER_POST = 50
# Maximum attempts to rotate the Tor exit node before giving up on a request
MAX_ROTATION_ATTEMPTS = 10


def _detect_tor_proxy():
    """
    Return a proxies dict for requests if a Tor SOCKS5 proxy is available,
    otherwise return None.

    Detection order:
      1. TOR_PROXY env var (e.g. ``socks5h://127.0.0.1:9050``)
      2. torsocks transparent-proxy mode (LD_PRELOAD set by torsocks)
      3. A reachable TCP socket on 127.0.0.1:9050 (default Tor daemon port)
    """
    env_proxy = os.environ.get('TOR_PROXY', '').strip()
    if env_proxy:
        return {'http': env_proxy, 'https': env_proxy}

    # Detect torsocks LD_PRELOAD wrapping – socket calls are transparently
    # redirected so we don't need an explicit proxy dict.
    if 'torsocks' in os.environ.get('LD_PRELOAD', '').lower():
        return {}  # empty dict → use default (torsocks handles it)

    # Try connecting directly to the default Tor SOCKS port
    try:
        s = socket.create_connection(('127.0.0.1', 9050), timeout=2)
        s.close()
        proxy_url = 'socks5h://127.0.0.1:9050'
        return {'http': proxy_url, 'https': proxy_url}
    except OSError:
        pass

    return None


# Determine proxies once at startup
_PROXIES = _detect_tor_proxy()
if _PROXIES is not None:
    proxy_desc = _PROXIES or 'torsocks LD_PRELOAD (transparent)'
    print(f"[tor] Routing requests through Tor ({proxy_desc})")
else:
    print("[info] No Tor proxy detected – using direct connection")


def rotate_tor_exit(max_wait: int = 120) -> bool:
    """
    Restart the Tor daemon to obtain a fresh exit node.

    Returns True if Tor bootstrapped successfully within *max_wait* seconds,
    False otherwise.  Safe to call even when Tor is not installed (returns
    False immediately).
    """
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'tor@default'],
                       capture_output=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False

    deadline = time.time() + max_wait
    while time.time() < deadline:
        time.sleep(3)
        try:
            result = subprocess.run(
                ['sudo', 'journalctl', '-u', 'tor@default',
                 '--no-pager', '-n', '30', '-r'],
                capture_output=True, text=True, timeout=10
            )
            if 'Bootstrapped 100%' in result.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    return False


def reddit_get(url, params=None):
    """
    Make a GET request to the Reddit JSON API.

    When Tor is active and Reddit returns 429 or 403, the exit node is rotated
    automatically and the request is retried (up to MAX_ROTATION_ATTEMPTS).
    For non-Tor connections a simple exponential back-off is used instead.
    """
    kwargs = dict(headers=HEADERS, params=params, timeout=30)
    if _PROXIES is not None:
        kwargs['proxies'] = _PROXIES

    for attempt in range(MAX_ROTATION_ATTEMPTS):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code == 200:
                return response.json()

            if response.status_code in (429, 403):
                if _PROXIES is not None:
                    # Tor is active – rotate exit node and retry immediately
                    print(f"  [tor] HTTP {response.status_code} – rotating exit node "
                          f"(attempt {attempt + 1}/{MAX_ROTATION_ATTEMPTS}) …")
                    if rotate_tor_exit():
                        print("  [tor] New exit node ready, retrying …")
                    else:
                        print("  [tor] Rotation failed, waiting 10 s …")
                        time.sleep(10)
                else:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited – waiting {wait}s …")
                    time.sleep(wait)
            else:
                print(f"  HTTP {response.status_code} for {url}")
                return None

        except Exception as exc:
            print(f"  Request error ({exc}), retrying …")
            time.sleep(5)

    print(f"  Gave up after {MAX_ROTATION_ATTEMPTS} attempts for {url}")
    return None


def fetch_listing(subreddit, listing='new', time_filter='all', after=None):
    """
    Fetch one page (up to LIMIT items) from a subreddit listing.

    Parameters
    ----------
    listing     : 'new' | 'top' | 'hot' | 'rising'
    time_filter : 'all' | 'year' | 'month' | 'week' | 'day'  (only for 'top')
    after       : fullname (e.g. 't3_abc123') for pagination cursor
    """
    url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
    params = {'limit': LIMIT, 'raw_json': 1}
    if listing == 'top':
        params['t'] = time_filter
    if after:
        params['after'] = after
    data = reddit_get(url, params=params)
    if data and 'data' in data:
        return data['data'].get('children', []), data['data'].get('after')
    return [], None


def collect_submissions(subreddit, max_pages=10):
    """
    Collect posts from a subreddit by paginating through:
      - 'new' listing  (gets the most recent ~1000 posts)
      - 'top?t=all'    (gets the highest-voted all-time posts – often older)

    Parameters
    ----------
    max_pages : maximum pages to fetch per listing (default 10 = ~1000 posts).
                Pass 1 for a quick seed run (~100 posts per listing).

    Returns a list of raw post dicts.
    """
    posts = {}

    for listing, tf in [('new', None), ('top', 'all')]:
        after = None
        page = 0
        label = f"r/{subreddit}/{listing}"
        print(f"  Fetching {label} …")
        while True:
            children, after = fetch_listing(subreddit, listing=listing,
                                            time_filter=tf or 'all', after=after)
            if not children:
                break
            for child in children:
                post = child.get('data')
                if not isinstance(post, dict):
                    continue
                post_id = post.get('id')
                if not post_id:
                    continue
                posts[post_id] = post
            page += 1
            time.sleep(REQUEST_DELAY)
            if not after or page >= max_pages:
                break

    return list(posts.values())


def fetch_post_comments(subreddit, post_id):
    """
    Fetch top-level comments for a single post.
    Returns a flat list of comment dicts.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = reddit_get(url, params={'limit': MAX_COMMENTS_PER_POST, 'depth': 1, 'raw_json': 1})
    if not data or len(data) < 2:
        return []

    comments = []
    for child in data[1]['data'].get('children', []):
        if child.get('kind') == 't1':
            comments.append(child['data'])
    return comments


def _hash_author(username):
    """
    Return a stable 16-char hex hash of a Reddit username.

    Hashing avoids committing raw usernames to the repository while still
    preserving uniqueness (for counting unique contributors).
    Returns an empty string for deleted/unknown accounts.
    """
    if not username or username in ('[deleted]', '[removed]'):
        return ''
    return hashlib.sha256(username.encode()).hexdigest()[:16]


def _utc_date(ts):
    """Convert a UTC Unix timestamp to a YYYY-MM-DD string."""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
    except (OSError, OverflowError, ValueError):
        return ''


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


def _load_existing_ids_from_zst(data_dir: Path, subreddit: str, kind: str) -> set:
    """
    Load existing IDs from zst archives in data/ folder.

    Parameters
    ----------
    data_dir : Path to data directory containing zst archives
    subreddit : Name of subreddit (case-insensitive)
    kind : 'submissions' or 'comments'

    Returns set of IDs already present in the archive.
    """
    zst_file = data_dir / f"{subreddit}_{kind}.zst"
    if not zst_file.exists():
        return set()

    existing_ids = set()
    for obj in _iter_ndjson_zst(zst_file):
        obj_id = obj.get('id')
        if obj_id:
            existing_ids.add(obj_id)
    return existing_ids


def _append_to_zst(data_dir: Path, subreddit: str, kind: str, new_items: list):
    """
    Append new items to a zst archive, deduplicating by ID.

    Parameters
    ----------
    data_dir : Path to data directory containing zst archives
    subreddit : Name of subreddit
    kind : 'submissions' or 'comments'
    new_items : List of dicts to append

    Returns the number of truly new items added.
    """
    if not new_items:
        return 0

    data_dir.mkdir(parents=True, exist_ok=True)
    zst_file = data_dir / f"{subreddit}_{kind}.zst"

    # Load existing items
    existing_items = {}
    if zst_file.exists():
        for obj in _iter_ndjson_zst(zst_file):
            obj_id = obj.get('id')
            if obj_id:
                existing_items[obj_id] = obj

    # Add new items, deduplicating
    initial_count = len(existing_items)
    for item in new_items:
        item_id = item.get('id')
        if item_id:
            existing_items[item_id] = item

    # Write all items back to the zst file
    cctx = zstandard.ZstdCompressor()
    with open(zst_file, 'wb') as fh:
        with cctx.stream_writer(fh) as writer:
            for item in existing_items.values():
                line = json.dumps(item) + '\n'
                writer.write(line.encode('utf-8'))

    return len(existing_items) - initial_count


def extract_submission_features(post):
    """Return a flat dict of relevant submission fields."""
    created = post.get('created_utc', 0)
    return {
        'id': post.get('id'),
        'subreddit': post.get('subreddit'),
        'title': post.get('title', ''),
        'selftext': post.get('selftext', ''),
        'author_hash': _hash_author(post.get('author')),
        'score': post.get('score', 0),
        'upvote_ratio': post.get('upvote_ratio', None),
        'num_comments': post.get('num_comments', 0),
        'created_utc': created,
        'created_date': _utc_date(created),
        'url': post.get('url', ''),
        'is_self': post.get('is_self', False),
        'permalink': post.get('permalink', ''),
    }


def extract_comment_features(comment, subreddit):
    """Return a flat dict of relevant comment fields."""
    created = comment.get('created_utc', 0)
    return {
        'id': comment.get('id'),
        'subreddit': subreddit,
        'body': comment.get('body', ''),
        'author_hash': _hash_author(comment.get('author')),
        'score': comment.get('score', 0),
        'created_utc': created,
        'created_date': _utc_date(created),
        'parent_id': comment.get('parent_id', ''),
        'link_id': comment.get('link_id', ''),
    }


def main():
    parser = argparse.ArgumentParser(description='Collect Reddit AuDHD data')
    parser.add_argument(
        '--seed', action='store_true',
        help='Seed mode: fetch only 1 page per listing and skip comments. '
             'Useful for a quick proof-of-concept run.'
    )
    parser.add_argument(
        '--data-dir', default='data',
        help='Directory containing zst archives (default: data/)'
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.seed:
        print("=== SEED MODE: 1 page per listing, no comments ===")
        max_pages = 1
    else:
        max_pages = 10  # Reddit caps at ~1000 posts (10 × 100)

    print("Reddit data collection – official JSON API")
    print(f"Target subreddits: {', '.join(ALL_SUBREDDITS)}")
    print(f"Data directory: {data_dir}")
    print()

    total_new_submissions = 0
    total_new_comments = 0

    for subreddit in ALL_SUBREDDITS:
        print(f"\n{'='*60}")
        print(f"Subreddit: r/{subreddit}")
        print('='*60)

        # Load existing IDs from zst archive
        print(f"  Loading existing submission IDs from archive...")
        existing_submission_ids = _load_existing_ids_from_zst(data_dir, subreddit, 'submissions')
        print(f"  → {len(existing_submission_ids)} existing submissions in archive")

        subreddit_submissions = []

        # --- Submissions ---
        posts = collect_submissions(subreddit, max_pages=max_pages)
        print(f"  → {len(posts)} unique posts collected from API")

        # Filter to only new posts not in archive
        new_posts = [p for p in posts if p.get('id') not in existing_submission_ids]
        print(f"  → {len(new_posts)} new posts (not in archive)")

        sub_rows = [extract_submission_features(p) for p in new_posts]
        subreddit_submissions.extend(sub_rows)

        if not args.seed and new_posts:
            # --- Comments (fetch per post) ---
            # Load existing comment IDs
            print(f"  Loading existing comment IDs from archive...")
            existing_comment_ids = _load_existing_ids_from_zst(data_dir, subreddit, 'comments')
            print(f"  → {len(existing_comment_ids)} existing comments in archive")

            # Build map of existing comments by link_id
            existing_comment_links = {}
            zst_file = data_dir / f"{subreddit}_comments.zst"
            if zst_file.exists():
                for comment in _iter_ndjson_zst(zst_file):
                    link_id = comment.get('link_id', '')
                    if link_id:
                        existing_comment_links[link_id] = existing_comment_links.get(link_id, 0) + 1

            posts_to_fetch = []
            skipped_count = 0
            for post in new_posts:
                post_id = post['id']
                link_id = f"t3_{post_id}"
                num_comments = post.get('num_comments', 0)
                collected_comments = existing_comment_links.get(link_id, 0)

                # Skip if post has no comments or we already have all comments
                if num_comments == 0:
                    skipped_count += 1
                    continue
                if collected_comments >= min(num_comments, MAX_COMMENTS_PER_POST):
                    skipped_count += 1
                    continue

                posts_to_fetch.append(post)

            if skipped_count > 0:
                print(f"  → Skipping {skipped_count} posts (no comments or already collected)")

            subreddit_comments = []
            print(f"  Fetching comments for {len(posts_to_fetch)} posts …")
            # Save comments every SAVE_INTERVAL posts to prevent data loss on timeout
            SAVE_INTERVAL = 50
            checkpoint_new_comments = 0
            for idx, post in enumerate(tqdm(posts_to_fetch, desc=f"  r/{subreddit} comments"), start=1):
                comments = fetch_post_comments(subreddit, post['id'])
                for c in comments:
                    subreddit_comments.append(extract_comment_features(c, subreddit))
                time.sleep(REQUEST_DELAY)

                # Save incrementally every SAVE_INTERVAL posts to survive timeouts
                if idx % SAVE_INTERVAL == 0 and subreddit_comments:
                    saved_coms = _append_to_zst(data_dir, subreddit, 'comments', subreddit_comments)
                    checkpoint_new_comments += saved_coms
                    if saved_coms > 0:
                        print(f"\n  → Checkpoint: saved {saved_coms} new comments ({idx}/{len(posts_to_fetch)} posts processed)")
                    # Clear the buffer after saving
                    subreddit_comments = []

            print(f"  → {len(subreddit_comments)} comments collected in final batch")
            total_new_comments += checkpoint_new_comments

            # Save final batch of comments
            if subreddit_comments:
                saved_coms = _append_to_zst(data_dir, subreddit, 'comments', subreddit_comments)
                total_new_comments += saved_coms
        elif not args.seed:
            print(f"  → No new posts to fetch comments for")

        # --- Save submissions to zst archive ---
        if subreddit_submissions:
            new_subs = _append_to_zst(data_dir, subreddit, 'submissions', subreddit_submissions)
            total_new_submissions += new_subs
            print(f"  → Saved {new_subs} new submissions to archive")
        else:
            print(f"  → No new submissions to save")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"New submissions this run : {total_new_submissions}")
    print(f"New comments this run    : {total_new_comments}")

    # Count total items across all archives
    total_subs = 0
    total_coms = 0
    for subreddit in ALL_SUBREDDITS:
        sub_ids = _load_existing_ids_from_zst(data_dir, subreddit, 'submissions')
        com_ids = _load_existing_ids_from_zst(data_dir, subreddit, 'comments')
        total_subs += len(sub_ids)
        total_coms += len(com_ids)

    print(f"Total submissions (all)  : {total_subs}")
    print(f"Total comments (all)     : {total_coms}")
    print(f"\nFiles written to: {data_dir}/")


if __name__ == "__main__":
    main()
