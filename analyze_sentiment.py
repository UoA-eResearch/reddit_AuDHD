#!/usr/bin/env python3
"""
Script to analyze sentiment of Reddit posts and comments about Autism and ADHD.

Data sources:
1. Historical seed data: zst-compressed archives in data/ folder (from academictorrents.com)
2. 2026 data: CSV files (reddit_submissions_2026.csv, reddit_comments_2026.csv)

Combines both sources, deduplicates by ID, and performs sentiment analysis.

Memory-efficient streaming design:
- Processes data one batch at a time (50k rows)
- Runs sentiment analysis inline per batch (single-threaded VADER)
- Writes results to CSV incrementally and frees each batch before the next
- Keeps peak memory bounded to ~1 batch regardless of total dataset size
- Large text columns (body, selftext) excluded from output to manage file size;
  original text is preserved in source zst/csv files
"""

import hashlib
import json
import gc
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zstandard
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# LFS pointer files start with this prefix
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com"
# Zstd magic bytes (first 4 bytes of any valid zstd frame)
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _hash_author(username: str) -> str:
    """Return first 16 hex chars of SHA-256(username), empty for deleted."""
    if not username or username in ("[deleted]", "[removed]"):
        return ""
    return hashlib.sha256(username.encode()).hexdigest()[:16]


def _utc_date(ts) -> str:
    """Convert a UTC timestamp to a YYYY-MM-DD string."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyze sentiment of a text using VADER.
    Returns compound score (-1 to 1, negative to positive).
    """
    if pd.isna(text) or text == '' or text == '[deleted]' or text == '[removed]':
        return None

    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

def categorize_sentiment(compound_score):
    """
    Categorize sentiment based on compound score.
    """
    if compound_score is None:
        return 'neutral'
    elif compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def _iter_ndjson_zst(zst_path: Path):
    """Yield parsed JSON objects from a zstd-compressed NDJSON file.

    Raises RuntimeError with a helpful message if the file looks like a Git LFS
    pointer (i.e., `git lfs pull` has not been run) or is not a valid zstd archive.
    """
    with open(zst_path, "rb") as fh:
        header = fh.read(len(_LFS_POINTER_PREFIX))

    if header.startswith(_LFS_POINTER_PREFIX):
        raise RuntimeError(
            f"{zst_path} is a Git LFS pointer file, not the real archive. "
            "Run `git lfs pull` to download the actual data."
        )
    if not header.startswith(_ZSTD_MAGIC):
        raise RuntimeError(
            f"{zst_path} does not appear to be a valid zstd archive "
            f"(first bytes: {header[:4]!r}). Run `git lfs pull` if using Git LFS."
        )

    dctx = zstandard.ZstdDecompressor()
    buf = bytearray()
    with open(zst_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        while True:
            chunk = reader.read(131_072)
            if not chunk:
                break
            buf.extend(chunk)
            newline_index = buf.find(b"\n")
            while newline_index != -1:
                line = buf[:newline_index].strip()
                del buf[:newline_index + 1]
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
                newline_index = buf.find(b"\n")
    remaining = buf.strip()
    if remaining:
        try:
            yield json.loads(remaining)
        except json.JSONDecodeError:
            pass


def _dataframe_batches(records, columns, batch_size=50_000):
    """Yield DataFrames built from bounded-size batches of record dicts."""
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield pd.DataFrame(batch, columns=columns)
            batch.clear()
    if batch:
        yield pd.DataFrame(batch, columns=columns)


# ---- Module-level constants ------------------------------------------------

_AUTISM_SUBREDDITS = frozenset({'autism', 'aspergers', 'aspergirls', 'autisticadults'})

_SUB_COLUMNS = [
    'id', 'subreddit', 'title', 'selftext', 'author_hash', 'score',
    'upvote_ratio', 'num_comments', 'created_utc', 'created_date',
    'url', 'is_self', 'permalink',
]

_COM_COLUMNS = [
    'id', 'subreddit', 'body', 'author_hash', 'score',
    'created_utc', 'created_date', 'parent_id', 'link_id',
]

# Output columns -- large text columns (selftext, body) are excluded to keep
# output CSVs at a manageable size.  The original text is always available in
# the source zst archives and 2026 CSV files.
_SUB_OUTPUT_COLUMNS = [
    'id', 'subreddit', 'title', 'author_hash', 'score',
    'upvote_ratio', 'num_comments', 'created_utc', 'created_date',
    'url', 'is_self', 'permalink',
    'sentiment_score', 'sentiment_category', 'category',
]

_COM_OUTPUT_COLUMNS = [
    'id', 'subreddit', 'author_hash', 'score',
    'created_utc', 'created_date',
    'sentiment_score', 'sentiment_category', 'category',
]


# ---- Memory / disk helpers -------------------------------------------------

def _log_memory(label: str):
    """Log process RSS and system memory usage (Linux /proc)."""
    pid = os.getpid()
    rss_kb = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
    except (FileNotFoundError, ValueError):
        pass
    total_kb = avail_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except (FileNotFoundError, ValueError):
        pass
    used_kb = total_kb - avail_kb
    pct = (used_kb * 100 / total_kb) if total_kb > 0 else 0
    print(f"[Memory] {label}: Process RSS={rss_kb / 1048576:.2f}GiB | "
          f"System: {used_kb / 1048576:.2f}/{total_kb / 1048576:.2f}GiB ({pct:.1f}%)")


def _check_disk(path: str = "/"):
    """Return (free_gb, total_gb) for the filesystem containing *path*."""
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize / (1024 ** 3)
    total = st.f_blocks * st.f_frsize / (1024 ** 3)
    return free, total


# ---- Record extraction helpers (module-level for clarity) ------------------

def _submission_record(post):
    """Extract a flat dict from a raw submission JSON object."""
    post_id = post.get('id', '')
    if not post_id:
        return None
    ts = post.get('created_utc', 0)
    return {
        'id': post_id,
        'subreddit': post.get('subreddit', ''),
        'title': post.get('title', ''),
        'selftext': post.get('selftext', ''),
        'author_hash': _hash_author(post.get('author', '')),
        'score': post.get('score', 0),
        'upvote_ratio': post.get('upvote_ratio', None),
        'num_comments': post.get('num_comments', 0),
        'created_utc': ts,
        'created_date': _utc_date(ts),
        'url': post.get('url', ''),
        'is_self': post.get('is_self', False),
        'permalink': post.get('permalink', ''),
    }


def _comment_record(comment):
    """Extract a flat dict from a raw comment JSON object."""
    comment_id = comment.get('id', '')
    if not comment_id:
        return None
    ts = comment.get('created_utc', 0)
    return {
        'id': comment_id,
        'subreddit': comment.get('subreddit', ''),
        'body': comment.get('body', ''),
        'author_hash': _hash_author(comment.get('author', '')),
        'score': comment.get('score', 0),
        'created_utc': ts,
        'created_date': _utc_date(ts),
        'parent_id': comment.get('parent_id', ''),
        'link_id': comment.get('link_id', ''),
    }


# ---- Batch enrichment helpers ----------------------------------------------

def _enrich_submission_batch(df):
    """Add sentiment scores and category to a submission DataFrame in-place.

    The selftext column is dropped after use to reduce memory and output size.
    """
    text = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    df['sentiment_score'] = text.apply(analyze_sentiment)
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    df['category'] = df['subreddit'].apply(
        lambda s: 'Autism' if s.lower() in _AUTISM_SUBREDDITS else 'ADHD'
    )
    df.drop(columns=['selftext'], inplace=True, errors='ignore')


def _enrich_comment_batch(df):
    """Add sentiment scores and category to a comment DataFrame in-place.

    The body column is dropped after use to reduce memory and output size.
    """
    df['sentiment_score'] = df['body'].apply(analyze_sentiment)
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    df['category'] = df['subreddit'].apply(
        lambda s: 'Autism' if s.lower() in _AUTISM_SUBREDDITS else 'ADHD'
    )
    df.drop(columns=['body'], inplace=True, errors='ignore')


# ---- Streaming pipeline ----------------------------------------------------

def _stream_zst_to_csv(data_dir, glob_pattern, record_fn, columns,
                       enrich_fn, output_columns, output_path,
                       exclude_ids, label="records"):
    """Stream zst files → sentiment analysis → incremental CSV output.

    Each batch of 50 000 records is enriched (sentiment + category) and
    immediately written to *output_path* before being freed, so peak memory
    is bounded to a single batch regardless of total dataset size.

    Returns (total_rows_written, header_already_written).
    """
    first_write = True
    total_rows = 0

    for zst_file in sorted(data_dir.glob(glob_pattern)):
        print(f"  Processing {zst_file.name}...")
        file_rows = 0
        batch_count = 0

        records = (r for obj in _iter_ndjson_zst(zst_file)
                   if (r := record_fn(obj)) is not None
                   and r['id'] not in exclude_ids)

        for batch_df in _dataframe_batches(records, columns):
            enrich_fn(batch_df)

            out_cols = [c for c in output_columns if c in batch_df.columns]
            batch_df[out_cols].to_csv(
                output_path,
                mode='w' if first_write else 'a',
                header=first_write,
                index=False,
            )
            first_write = False
            file_rows += len(batch_df)
            batch_count += 1
            del batch_df

            # Periodic progress every 10 batches (~500k rows)
            if batch_count % 10 == 0:
                _log_memory(f"  {zst_file.stem}: {file_rows:,} {label}")

        total_rows += file_rows
        print(f"    → {file_rows:,} {label}")
        _log_memory(f"After {zst_file.name}")
        gc.collect()

    return total_rows, first_write


def load_and_analyze_data():
    """Load historical + 2026 data, run sentiment analysis, write output CSVs.

    Uses a streaming pipeline so that at most one batch (50 000 rows) of raw
    data is held in memory at any time.  After writing the output CSVs, a
    lightweight re-read (selected columns, categorical dtypes) provides the
    DataFrames needed for visualisation and statistics.

    Returns (submissions_df, comments_df) with columns sufficient for
    create_visualizations() and generate_statistics().
    """
    data_dir = Path("data")
    sub_output = Path("reddit_submissions_with_sentiment_2026.csv")
    com_output = Path("reddit_comments_with_sentiment_2026.csv")
    csv_2026_subs = Path("reddit_submissions_2026.csv")
    csv_2026_coms = Path("reddit_comments_2026.csv")

    _log_memory("Start")
    free_gb, total_gb = _check_disk()
    print(f"[Disk] Free: {free_gb:.1f}GiB / {total_gb:.1f}GiB")

    # ================================================================
    # Phase 1 – Submissions
    # ================================================================
    print("\n=== Phase 1: Processing submissions ===")

    # Pre-load 2026 submission IDs for deduplication
    sub_exclude_ids: set = set()
    if csv_2026_subs.exists():
        try:
            sub_exclude_ids = set(pd.read_csv(csv_2026_subs, usecols=['id'])['id'])
            print(f"  Loaded {len(sub_exclude_ids):,} IDs from 2026 CSV for dedup")
        except (pd.errors.EmptyDataError, KeyError):
            pass

    sub_rows, sub_first_write = _stream_zst_to_csv(
        data_dir, "*_submissions.zst",
        _submission_record, _SUB_COLUMNS,
        _enrich_submission_batch, _SUB_OUTPUT_COLUMNS,
        sub_output, sub_exclude_ids, "submissions",
    )

    # Append 2026 CSV submissions (with sentiment)
    if csv_2026_subs.exists():
        print("  Processing 2026 submissions CSV...")
        try:
            subs_2026 = pd.read_csv(csv_2026_subs)
            if not subs_2026.empty:
                _enrich_submission_batch(subs_2026)
                out_cols = [c for c in _SUB_OUTPUT_COLUMNS if c in subs_2026.columns]
                subs_2026[out_cols].to_csv(
                    sub_output,
                    mode='w' if sub_first_write else 'a',
                    header=sub_first_write,
                    index=False,
                )
                sub_first_write = False
                print(f"    → {len(subs_2026):,} submissions from 2026 CSV")
                sub_rows += len(subs_2026)
                del subs_2026
        except pd.errors.EmptyDataError:
            pass

    gc.collect()
    _log_memory("After all submissions")
    print(f"  Total submissions written: {sub_rows:,}")

    # ================================================================
    # Phase 2 – Comments
    # ================================================================
    print("\n=== Phase 2: Processing comments ===")

    com_exclude_ids: set = set()
    if csv_2026_coms.exists():
        try:
            com_exclude_ids = set(pd.read_csv(csv_2026_coms, usecols=['id'])['id'])
            print(f"  Loaded {len(com_exclude_ids):,} IDs from 2026 CSV for dedup")
        except (pd.errors.EmptyDataError, KeyError):
            pass

    com_rows, com_first_write = _stream_zst_to_csv(
        data_dir, "*_comments.zst",
        _comment_record, _COM_COLUMNS,
        _enrich_comment_batch, _COM_OUTPUT_COLUMNS,
        com_output, com_exclude_ids, "comments",
    )

    # Append 2026 CSV comments (with sentiment)
    if csv_2026_coms.exists():
        print("  Processing 2026 comments CSV...")
        try:
            coms_2026 = pd.read_csv(csv_2026_coms)
            if not coms_2026.empty:
                _enrich_comment_batch(coms_2026)
                out_cols = [c for c in _COM_OUTPUT_COLUMNS if c in coms_2026.columns]
                coms_2026[out_cols].to_csv(
                    com_output,
                    mode='w' if com_first_write else 'a',
                    header=com_first_write,
                    index=False,
                )
                com_first_write = False
                print(f"    → {len(coms_2026):,} comments from 2026 CSV")
                com_rows += len(coms_2026)
                del coms_2026
        except pd.errors.EmptyDataError:
            pass

    gc.collect()
    _log_memory("After all comments")
    print(f"  Total comments written: {com_rows:,}")

    # ================================================================
    # Phase 3 – Re-read lightweight DataFrames for visualisation / stats
    # ================================================================
    print("\n=== Phase 3: Loading results for visualisation ===")
    _log_memory("Before viz re-read")

    if sub_rows > 0:
        submissions_df = pd.read_csv(
            sub_output,
            usecols=['subreddit', 'title', 'sentiment_score',
                     'sentiment_category', 'created_date'],
            dtype={'subreddit': 'category', 'sentiment_category': 'category'},
        )
        submissions_df['created_date'] = pd.to_datetime(
            submissions_df['created_date'], errors='coerce')
        submissions_df['category'] = submissions_df['subreddit'].apply(
            lambda s: 'Autism' if str(s).lower() in _AUTISM_SUBREDDITS else 'ADHD'
        ).astype('category')
    else:
        submissions_df = pd.DataFrame()

    _log_memory("Submissions loaded for viz")

    if com_rows > 0:
        comments_df = pd.read_csv(
            com_output,
            usecols=['subreddit', 'sentiment_score',
                     'sentiment_category', 'created_date'],
            dtype={'subreddit': 'category', 'sentiment_category': 'category'},
        )
        comments_df['created_date'] = pd.to_datetime(
            comments_df['created_date'], errors='coerce')
        comments_df['category'] = comments_df['subreddit'].apply(
            lambda s: 'Autism' if str(s).lower() in _AUTISM_SUBREDDITS else 'ADHD'
        ).astype('category')
    else:
        comments_df = pd.DataFrame()

    _log_memory("Comments loaded for viz")

    if submissions_df.empty:
        print("WARNING: No submissions found in data/ folder or CSV files")

    return submissions_df, comments_df

def create_visualizations(submissions_df, comments_df):
    """
    Create visualizations for sentiment analysis.
    """
    print("\nCreating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    # Fixed sentiment order and color mapping so chart colors are always semantically correct
    SENTIMENT_ORDER = ['negative', 'neutral', 'positive']
    SENTIMENT_COLORS = {'negative': '#ff9999', 'neutral': '#66b3ff', 'positive': '#99ff99'}

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Reddit Sentiment Analysis: Autism and ADHD Communities', fontsize=16, fontweight='bold')

    # 1. Sentiment distribution for submissions
    sentiment_counts = submissions_df['sentiment_category'].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=[SENTIMENT_COLORS[label] for label in sentiment_counts.index])
    axes[0, 0].set_title('Submission Sentiment Distribution')

    # 2. Sentiment distribution for comments
    if not comments_df.empty and 'sentiment_category' in comments_df.columns:
        comment_sentiment_counts = comments_df['sentiment_category'].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        axes[0, 1].pie(comment_sentiment_counts.values, labels=comment_sentiment_counts.index, autopct='%1.1f%%',
                       colors=[SENTIMENT_COLORS[label] for label in comment_sentiment_counts.index])
    else:
        axes[0, 1].text(0.5, 0.5, 'No comments collected yet', ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Comment Sentiment Distribution')

    # 3. Sentiment over time (monthly average) for submissions
    submissions_monthly = submissions_df.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
    axes[1, 0].plot(submissions_monthly.index, submissions_monthly.values, marker='o', linewidth=2)
    axes[1, 0].set_title('Average Submission Sentiment Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Average Sentiment Score')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Sentiment over time (monthly average) for comments
    if not comments_df.empty and 'sentiment_score' in comments_df.columns:
        comments_monthly = comments_df.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
        axes[1, 1].plot(comments_monthly.index, comments_monthly.values, marker='o', linewidth=2, color='orange')
    else:
        axes[1, 1].text(0.5, 0.5, 'No comments collected yet', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Average Comment Sentiment Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Average Sentiment Score')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Sentiment by category (Autism vs ADHD) - Submissions
    category_sentiment_sub = submissions_df.groupby('category')['sentiment_score'].mean()
    axes[2, 0].bar(category_sentiment_sub.index, category_sentiment_sub.values, color=['#8B4789', '#4A90E2'])
    axes[2, 0].set_title('Average Submission Sentiment by Category')
    axes[2, 0].set_ylabel('Average Sentiment Score')
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].grid(True, alpha=0.3, axis='y')

    # 6. Sentiment by category (Autism vs ADHD) - Comments
    if not comments_df.empty and 'category' in comments_df.columns and 'sentiment_score' in comments_df.columns:
        category_sentiment_com = comments_df.groupby('category')['sentiment_score'].mean()
        axes[2, 1].bar(category_sentiment_com.index, category_sentiment_com.values, color=['#8B4789', '#4A90E2'])
    else:
        axes[2, 1].text(0.5, 0.5, 'No comments collected yet', ha='center', va='center', transform=axes[2, 1].transAxes)
    axes[2, 1].set_title('Average Comment Sentiment by Category')
    axes[2, 1].set_ylabel('Average Sentiment Score')
    axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('sentiment_analysis_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_analysis_overview.png")
    plt.close()

    # Additional plot: Sentiment over time by category
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Sentiment Over Time by Category', fontsize=16, fontweight='bold')

    # Submissions by category
    for category in ['Autism', 'ADHD']:
        data = submissions_df[submissions_df['category'] == category]
        monthly = data.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
        axes[0].plot(monthly.index, monthly.values, marker='o', linewidth=2, label=category)

    axes[0].set_title('Submission Sentiment Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Average Sentiment Score')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Comments by category
    if not comments_df.empty and 'category' in comments_df.columns:
        for category in ['Autism', 'ADHD']:
            data = comments_df[comments_df['category'] == category]
            monthly = data.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
            axes[1].plot(monthly.index, monthly.values, marker='o', linewidth=2, label=category)
    else:
        axes[1].text(0.5, 0.5, 'No comments collected yet', ha='center', va='center', transform=axes[1].transAxes)

    axes[1].set_title('Comment Sentiment Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Average Sentiment Score')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sentiment_by_category.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_by_category.png")
    plt.close()

    # Subreddit comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Average Sentiment by Subreddit', fontsize=16, fontweight='bold')

    # Submissions
    sub_by_subreddit = submissions_df.groupby('subreddit')['sentiment_score'].mean().sort_values()
    axes[0].barh(sub_by_subreddit.index, sub_by_subreddit.values, color='steelblue')
    axes[0].set_xlabel('Average Sentiment Score')
    axes[0].set_title('Submissions')
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Comments
    if not comments_df.empty and 'sentiment_score' in comments_df.columns:
        com_by_subreddit = comments_df.groupby('subreddit')['sentiment_score'].mean().sort_values()
        axes[1].barh(com_by_subreddit.index, com_by_subreddit.values, color='coral')
    else:
        axes[1].text(0.5, 0.5, 'No comments collected yet', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('Average Sentiment Score')
    axes[1].set_title('Comments')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('sentiment_by_subreddit.png', dpi=300, bbox_inches='tight')
    print("Saved: sentiment_by_subreddit.png")
    plt.close()

def generate_statistics(submissions_df, comments_df):
    """
    Generate and print summary statistics.
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS STATISTICS")
    print("="*60)

    print("\nSUBMISSIONS:")
    print(f"Total submissions analyzed: {len(submissions_df)}")
    print(f"Average sentiment score: {submissions_df['sentiment_score'].mean():.4f}")
    print(f"Median sentiment score: {submissions_df['sentiment_score'].median():.4f}")
    print(f"Standard deviation: {submissions_df['sentiment_score'].std():.4f}")
    print("\nSentiment distribution:")
    print(submissions_df['sentiment_category'].value_counts())
    print("\nBy category:")
    print(submissions_df.groupby('category')['sentiment_score'].agg(['count', 'mean', 'std']))

    print("\n" + "-"*60)
    print("\nCOMMENTS:")
    print(f"Total comments analyzed: {len(comments_df)}")
    if not comments_df.empty and 'sentiment_score' in comments_df.columns:
        print(f"Average sentiment score: {comments_df['sentiment_score'].mean():.4f}")
        print(f"Median sentiment score: {comments_df['sentiment_score'].median():.4f}")
        print(f"Standard deviation: {comments_df['sentiment_score'].std():.4f}")
        print("\nSentiment distribution:")
        print(comments_df['sentiment_category'].value_counts())
        print("\nBy category:")
        print(comments_df.groupby('category')['sentiment_score'].agg(['count', 'mean', 'std']))
    else:
        print("(No comments collected in this run)")

    # Time range
    print("\n" + "-"*60)
    print("\nTIME RANGE:")
    print(f"Submissions: {submissions_df['created_date'].min()} to {submissions_df['created_date'].max()}")
    if not comments_df.empty and 'created_date' in comments_df.columns and comments_df['created_date'].notna().any():
        print(f"Comments: {comments_df['created_date'].min()} to {comments_df['created_date'].max()}")
    else:
        print("Comments: N/A")

    # Most positive and negative submissions
    print("\n" + "-"*60)
    print("\nMOST POSITIVE SUBMISSIONS:")
    top_positive = submissions_df.nlargest(5, 'sentiment_score')[['subreddit', 'title', 'sentiment_score', 'created_date']]
    for idx, row in top_positive.iterrows():
        print(f"\nSubreddit: r/{row['subreddit']}")
        print(f"Title: {row['title'][:100]}...")
        print(f"Sentiment: {row['sentiment_score']:.4f}")
        print(f"Date: {row['created_date']}")

    print("\n" + "-"*60)
    print("\nMOST NEGATIVE SUBMISSIONS:")
    top_negative = submissions_df.nsmallest(5, 'sentiment_score')[['subreddit', 'title', 'sentiment_score', 'created_date']]
    for idx, row in top_negative.iterrows():
        print(f"\nSubreddit: r/{row['subreddit']}")
        print(f"Title: {row['title'][:100]}...")
        print(f"Sentiment: {row['sentiment_score']:.4f}")
        print(f"Date: {row['created_date']}")

    return {
        'submissions': {
            'total': len(submissions_df),
            'avg_sentiment': submissions_df['sentiment_score'].mean(),
            'median_sentiment': submissions_df['sentiment_score'].median(),
            'std_sentiment': submissions_df['sentiment_score'].std(),
            'sentiment_distribution': submissions_df['sentiment_category'].value_counts().to_dict(),
            'by_category': submissions_df.groupby('category')['sentiment_score'].mean().to_dict()
        },
        'comments': {
            'total': len(comments_df),
            'avg_sentiment': comments_df['sentiment_score'].mean() if not comments_df.empty else None,
            'median_sentiment': comments_df['sentiment_score'].median() if not comments_df.empty else None,
            'std_sentiment': comments_df['sentiment_score'].std() if not comments_df.empty else None,
            'sentiment_distribution': comments_df['sentiment_category'].value_counts().to_dict() if not comments_df.empty else {},
            'by_category': comments_df.groupby('category')['sentiment_score'].mean().to_dict() if not comments_df.empty else {}
        }
    }

def main():
    """
    Main function to orchestrate sentiment analysis.
    """
    print("Starting sentiment analysis...")
    _log_memory("Start of main")

    # Load, analyze, and write output CSVs (streaming pipeline)
    submissions_df, comments_df = load_and_analyze_data()

    # CSV files are already written by load_and_analyze_data()
    print("\nOutput CSVs written by streaming pipeline.")

    # Generate visualizations
    create_visualizations(submissions_df, comments_df)

    # Generate statistics
    stats = generate_statistics(submissions_df, comments_df)

    _log_memory("End of main")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("- reddit_submissions_with_sentiment_2026.csv")
    print("- reddit_comments_with_sentiment_2026.csv")
    print("- sentiment_analysis_overview.png")
    print("- sentiment_by_category.png")
    print("- sentiment_by_subreddit.png")

if __name__ == "__main__":
    main()
