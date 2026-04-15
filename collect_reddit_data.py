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
import os
import socket
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd
import requests
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


def _load_existing(path, cols):
    """Load an existing CSV, returning an empty DataFrame if absent/empty."""
    try:
        df = pd.read_csv(path)
        if df.empty or 'id' not in df.columns:
            return pd.DataFrame(columns=cols)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)


def _save_data_incrementally(new_submissions, new_comments, submission_cols, comment_cols):
    """
    Merge new data with existing CSV files and save immediately.
    This ensures data is preserved even if the script is interrupted by a timeout.

    Note: Data is saved to reddit_submissions_2026.csv and reddit_comments_2026.csv
    to clearly distinguish from the historical seed data in data/*.zst archives.
    """
    existing_submissions = _load_existing('reddit_submissions_2026.csv', submission_cols)
    existing_comments = _load_existing('reddit_comments_2026.csv', comment_cols)

    new_submissions_df = (pd.DataFrame(new_submissions, columns=submission_cols)
                          .drop_duplicates(subset='id') if new_submissions
                          else pd.DataFrame(columns=submission_cols))
    new_comments_df = (pd.DataFrame(new_comments, columns=comment_cols)
                       .drop_duplicates(subset='id') if new_comments
                       else pd.DataFrame(columns=comment_cols))

    # Count truly new items by finding IDs not in existing data
    existing_sub_ids = set(existing_submissions['id']) if not existing_submissions.empty else set()
    existing_com_ids = set(existing_comments['id']) if not existing_comments.empty else set()

    new_sub_ids = set(new_submissions_df['id']) if not new_submissions_df.empty else set()
    new_com_ids = set(new_comments_df['id']) if not new_comments_df.empty else set()

    truly_new_subs = len(new_sub_ids - existing_sub_ids)
    truly_new_coms = len(new_com_ids - existing_com_ids)

    submissions_df = (pd.concat([existing_submissions, new_submissions_df], ignore_index=True)
                      .drop_duplicates(subset='id'))
    comments_df = (pd.concat([existing_comments, new_comments_df], ignore_index=True)
                   .drop_duplicates(subset='id'))

    submissions_df.to_csv('reddit_submissions_2026.csv', index=False)
    comments_df.to_csv('reddit_comments_2026.csv', index=False)

    return truly_new_subs, truly_new_coms, len(submissions_df), len(comments_df)


def main():
    parser = argparse.ArgumentParser(description='Collect Reddit AuDHD data')
    parser.add_argument(
        '--seed', action='store_true',
        help='Seed mode: fetch only 1 page per listing and skip comments. '
             'Useful for a quick proof-of-concept run.'
    )
    args = parser.parse_args()

    if args.seed:
        print("=== SEED MODE: 1 page per listing, no comments ===")
        max_pages = 1
    else:
        max_pages = 10  # Reddit caps at ~1000 posts (10 × 100)

    print("Reddit data collection – official JSON API")
    print(f"Target subreddits: {', '.join(ALL_SUBREDDITS)}")
    print()

    # --- Schema for DataFrames (ensures CSV always has a header row) ---
    SUBMISSION_COLS = ['id', 'subreddit', 'title', 'selftext', 'author_hash', 'score',
                       'upvote_ratio', 'num_comments', 'created_utc', 'created_date',
                       'url', 'is_self', 'permalink']
    COMMENT_COLS = ['id', 'subreddit', 'body', 'author_hash', 'score',
                    'created_utc', 'created_date', 'parent_id', 'link_id']

    total_new_submissions = 0
    total_new_comments = 0

    for subreddit in ALL_SUBREDDITS:
        print(f"\n{'='*60}")
        print(f"Subreddit: r/{subreddit}")
        print('='*60)

        subreddit_submissions = []
        subreddit_comments = []

        # --- Submissions ---
        posts = collect_submissions(subreddit, max_pages=max_pages)
        print(f"  → {len(posts)} unique posts collected")
        sub_rows = [extract_submission_features(p) for p in posts]
        subreddit_submissions.extend(sub_rows)

        if not args.seed:
            # --- Comments (fetch per post) ---
            # Load existing comments to check what we already have
            existing_comments = _load_existing('reddit_comments.csv', COMMENT_COLS)
            existing_comment_links = {}
            if not existing_comments.empty:
                # Build a map of link_id -> count of comments we have
                for _, row in existing_comments.iterrows():
                    link_id = row.get('link_id', '')
                    if link_id:
                        existing_comment_links[link_id] = existing_comment_links.get(link_id, 0) + 1

            posts_to_fetch = []
            skipped_count = 0
            for post in posts:
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
                print(f"  → Skipping {skipped_count} posts (comments already collected)")

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
                    _, saved_coms, _, _ = _save_data_incrementally(
                        [], subreddit_comments, SUBMISSION_COLS, COMMENT_COLS
                    )
                    checkpoint_new_comments += saved_coms
                    if saved_coms > 0:
                        print(f"\n  → Checkpoint: saved {saved_coms} new comments ({idx}/{len(posts_to_fetch)} posts processed)")
                    # Clear the buffer after saving
                    subreddit_comments = []

            print(f"  → {len(subreddit_comments)} comments collected in final batch")
            total_new_comments += checkpoint_new_comments

        # --- Save incrementally after each subreddit ---
        new_subs, new_coms, total_subs, total_coms = _save_data_incrementally(
            subreddit_submissions, subreddit_comments, SUBMISSION_COLS, COMMENT_COLS
        )
        total_new_submissions += new_subs
        total_new_comments += new_coms
        print(f"  → Saved to CSV (total: {total_subs} submissions, {total_coms} comments)")

    # --- Load final saved data for summary ---
    submissions_df = _load_existing('reddit_submissions_2026.csv', SUBMISSION_COLS)
    comments_df = _load_existing('reddit_comments_2026.csv', COMMENT_COLS)

    if submissions_df.empty:
        print("\nWARNING: No submissions were collected.")
        print("This usually means the Reddit API is not accessible from this network.")
        print("Reddit blocks requests from datacenter/CI IP ranges.")
        print("To collect real data, run with Tor: TOR_PROXY=socks5h://127.0.0.1:9050 python3 collect_reddit_data.py")
        return

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"New submissions this run : {total_new_submissions}")
    print(f"New comments this run    : {total_new_comments}")
    print(f"Total submissions (all)  : {len(submissions_df)}")
    print(f"Total comments (all)     : {len(comments_df)}")
    print(f"Date range (posts)       : {submissions_df['created_date'].min()} → {submissions_df['created_date'].max()}")
    print(f"\nFiles written: reddit_submissions.csv  reddit_comments.csv")

    print("\nSubmissions by subreddit:")
    print(submissions_df['subreddit'].value_counts().to_string())
    if not comments_df.empty:
        print("\nComments by subreddit:")
        print(comments_df['subreddit'].value_counts().to_string())


if __name__ == "__main__":
    main()
