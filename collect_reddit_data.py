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

Tor routing
-----------
When the TOR_PROXY environment variable is set (or when a Tor SOCKS5 daemon
is detected on 127.0.0.1:9050), all requests are routed through Tor so that
Reddit's datacenter-IP blocks are bypassed.  The GHA workflow sets this
automatically via the "torsocks" wrapper; you can also run:

    torsocks python collect_reddit_data.py
    # or
    TOR_PROXY=socks5h://127.0.0.1:9050 python collect_reddit_data.py
"""

import os
import socket
import requests
import pandas as pd
import time
from datetime import datetime
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


def _detect_tor_proxy():
    """
    Return a proxies dict for requests if a Tor SOCKS5 proxy is available,
    otherwise return None.

    Detection order:
      1. TOR_PROXY env var (e.g. ``socks5h://127.0.0.1:9050``)
      2. A reachable TCP socket on 127.0.0.1:9050 (default Tor daemon port)
      3. torsocks transparent-proxy mode (LD_PRELOAD is set by torsocks)
    """
    env_proxy = os.environ.get('TOR_PROXY', '').strip()
    if env_proxy:
        return {'http': env_proxy, 'https': env_proxy}

    # Detect torsocks LD_PRELOAD wrapping – in that mode Python's socket calls
    # are transparently redirected, so we don't need an explicit proxy dict.
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


# Determine proxies once at import time
_PROXIES = _detect_tor_proxy()
if _PROXIES is not None:
    print(f"[tor] Routing requests through Tor SOCKS5 proxy (proxies={_PROXIES or 'torsocks LD_PRELOAD'})")
else:
    print("[info] No Tor proxy detected – using direct connection")


def reddit_get(url, params=None, retries=3):
    """Make a GET request to the Reddit JSON API with retry logic."""
    kwargs = dict(headers=HEADERS, params=params, timeout=30)
    if _PROXIES is not None:
        kwargs['proxies'] = _PROXIES
    for attempt in range(retries):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited – waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  HTTP {response.status_code} for {url}")
                return None
        except Exception as exc:
            print(f"  Request error ({exc}), retrying …")
            time.sleep(5)
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


def collect_submissions(subreddit):
    """
    Collect posts from a subreddit by paginating through:
      - 'new' listing  (gets the most recent ~1000 posts)
      - 'top?t=all'    (gets the highest-voted all-time posts – often older)

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
                post = child.get('data', {})
                posts[post['id']] = post
            page += 1
            time.sleep(REQUEST_DELAY)
            if not after or page >= 10:   # Reddit caps at ~1000 results (10 pages)
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


def extract_submission_features(post):
    """Return a flat dict of relevant submission fields."""
    created = post.get('created_utc', 0)
    return {
        'id': post.get('id'),
        'subreddit': post.get('subreddit'),
        'title': post.get('title', ''),
        'selftext': post.get('selftext', ''),
        'author': post.get('author'),
        'score': post.get('score', 0),
        'upvote_ratio': post.get('upvote_ratio', None),
        'num_comments': post.get('num_comments', 0),
        'created_utc': created,
        'created_date': datetime.utcfromtimestamp(created).strftime('%Y-%m-%d'),
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
        'author': comment.get('author'),
        'score': comment.get('score', 0),
        'created_utc': created,
        'created_date': datetime.utcfromtimestamp(created).strftime('%Y-%m-%d'),
        'parent_id': comment.get('parent_id', ''),
        'link_id': comment.get('link_id', ''),
    }


def main():
    print("Reddit data collection – official JSON API")
    print(f"Target subreddits: {', '.join(ALL_SUBREDDITS)}")
    print()

    all_submissions = []
    all_comments = []

    for subreddit in ALL_SUBREDDITS:
        print(f"\n{'='*60}")
        print(f"Subreddit: r/{subreddit}")
        print('='*60)

        # --- Submissions ---
        posts = collect_submissions(subreddit)
        print(f"  → {len(posts)} unique posts collected")
        sub_rows = [extract_submission_features(p) for p in posts]
        all_submissions.extend(sub_rows)

        # --- Comments (fetch per post) ---
        print(f"  Fetching comments for {len(posts)} posts …")
        for post in tqdm(posts, desc=f"  r/{subreddit} comments"):
            comments = fetch_post_comments(subreddit, post['id'])
            for c in comments:
                all_comments.append(extract_comment_features(c, subreddit))
            time.sleep(REQUEST_DELAY)

        print(f"  → {sum(1 for c in all_comments if c['subreddit'] == subreddit)} comments so far")

    # --- Persist ---
    submissions_df = pd.DataFrame(all_submissions).drop_duplicates(subset='id') if all_submissions else pd.DataFrame()
    comments_df = pd.DataFrame(all_comments).drop_duplicates(subset='id') if all_comments else pd.DataFrame()

    if submissions_df.empty:
        print("\nWARNING: No submissions were collected.")
        print("This usually means the Reddit API is not accessible from this network.")
        print("Reddit blocks requests from datacenter/CI IP ranges.")
        print("To collect real data, run this script from a personal machine or use OAuth.")
        return

    submissions_df.to_csv('reddit_submissions.csv', index=False)
    comments_df.to_csv('reddit_comments.csv', index=False)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Total submissions : {len(submissions_df)}")
    print(f"Total comments    : {len(comments_df)}")
    print(f"Date range (posts): {submissions_df['created_date'].min()} → {submissions_df['created_date'].max()}")
    print(f"\nFiles written: reddit_submissions.csv  reddit_comments.csv")

    print("\nSubmissions by subreddit:")
    print(submissions_df['subreddit'].value_counts().to_string())
    print("\nComments by subreddit:")
    print(comments_df['subreddit'].value_counts().to_string())


if __name__ == "__main__":
    main()
