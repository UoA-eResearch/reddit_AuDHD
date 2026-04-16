#!/usr/bin/env python3
"""
Script to analyze sentiment of Reddit posts and comments about Autism and ADHD.

Data sources:
1. Historical seed data: zst-compressed archives in data/ folder (from academictorrents.com)
2. 2026 data: CSV files (reddit_submissions_2026.csv, reddit_comments_2026.csv)

Combines both sources, deduplicates by ID, and performs sentiment analysis.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zstandard
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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


def load_submissions_from_zst(data_dir: Path):
    """Load all submissions from zst archives in data_dir.

    Derives author_hash (SHA-256 of author) and created_date (from created_utc)
    to match the schema used by collect_reddit_data.py / import_seed_data.py.
    Records with a missing or empty id are skipped.

    Processes batches incrementally to avoid OOM: concatenates every few batches
    instead of accumulating all batches from a file in memory.
    """
    columns = [
        'id', 'subreddit', 'title', 'selftext', 'author_hash', 'score',
        'upvote_ratio', 'num_comments', 'created_utc', 'created_date',
        'url', 'is_self', 'permalink',
    ]
    all_frames = []
    for zst_file in sorted(data_dir.glob("*_submissions.zst")):
        print(f"  Loading {zst_file.name}...")

        def _submission_record(post):
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

        records = (r for post in _iter_ndjson_zst(zst_file)
                   if (r := _submission_record(post)) is not None)

        # Concatenate every 5 batches to avoid accumulating too many in memory
        batch_group = []
        for batch_df in _dataframe_batches(records, columns):
            batch_group.append(batch_df)
            if len(batch_group) >= 5:
                merged = pd.concat(batch_group, ignore_index=True)
                all_frames.append(merged)
                batch_group.clear()
        # Handle remaining batches
        if batch_group:
            merged = pd.concat(batch_group, ignore_index=True)
            all_frames.append(merged)

    if not all_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(all_frames, ignore_index=True)


def load_comments_from_zst(data_dir: Path):
    """Load all comments from zst archives in data_dir.

    Derives author_hash (SHA-256 of author) and created_date (from created_utc)
    to match the schema used by collect_reddit_data.py / import_seed_data.py.
    Records with a missing or empty id are skipped.

    Processes batches incrementally to avoid OOM: concatenates every few batches
    instead of accumulating all batches from a file in memory.
    """
    columns = [
        'id', 'subreddit', 'body', 'author_hash', 'score',
        'created_utc', 'created_date', 'parent_id', 'link_id',
    ]
    all_frames = []
    for zst_file in sorted(data_dir.glob("*_comments.zst")):
        print(f"  Loading {zst_file.name}...")

        def _comment_record(comment):
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

        records = (r for comment in _iter_ndjson_zst(zst_file)
                   if (r := _comment_record(comment)) is not None)

        # Concatenate every 5 batches to avoid accumulating too many in memory
        batch_group = []
        for batch_df in _dataframe_batches(records, columns):
            batch_group.append(batch_df)
            if len(batch_group) >= 5:
                merged = pd.concat(batch_group, ignore_index=True)
                all_frames.append(merged)
                batch_group.clear()
        # Handle remaining batches
        if batch_group:
            merged = pd.concat(batch_group, ignore_index=True)
            all_frames.append(merged)

    if not all_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(all_frames, ignore_index=True)


def load_and_analyze_data():
    """
    Load data from zst archives (historical seed data) and 2026 CSV files (new data).
    Combines both sources and deduplicates by ID.
    """
    data_dir = Path("data")

    print("Loading submissions from zst archives (historical seed data)...")
    submissions_df = load_submissions_from_zst(data_dir)

    print("\nLoading comments from zst archives (historical seed data)...")
    comments_df = load_comments_from_zst(data_dir)

    # Also load 2026 data from CSV files if they exist
    csv_2026_subs = Path("reddit_submissions_2026.csv")
    csv_2026_coms = Path("reddit_comments_2026.csv")

    if csv_2026_subs.exists():
        print("\nLoading 2026 submissions from CSV...")
        try:
            subs_2026 = pd.read_csv(csv_2026_subs)
            if not subs_2026.empty:
                print(f"  Found {len(subs_2026)} submissions from 2026")
                submissions_df = pd.concat([submissions_df, subs_2026], ignore_index=True)
                submissions_df = submissions_df.drop_duplicates(subset='id', keep='last')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    if csv_2026_coms.exists():
        print("\nLoading 2026 comments from CSV...")
        try:
            coms_2026 = pd.read_csv(csv_2026_coms)
            if not coms_2026.empty:
                print(f"  Found {len(coms_2026)} comments from 2026")
                comments_df = pd.concat([comments_df, coms_2026], ignore_index=True)
                comments_df = comments_df.drop_duplicates(subset='id', keep='last')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

    if submissions_df.empty:
        print("ERROR: No submissions found in data/ folder or CSV files")
        return pd.DataFrame(), pd.DataFrame()

    # Analyze sentiment for submissions
    print("\nAnalyzing sentiment for submissions...")
    # Combine title and selftext for submissions
    submissions_df['text'] = submissions_df['title'].fillna('') + ' ' + submissions_df['selftext'].fillna('')
    _n_workers = os.cpu_count() or 1
    _n_sub = len(submissions_df)
    _chunksize_sub = max(1, _n_sub // _n_workers)
    submissions_df['sentiment_score'] = process_map(
        analyze_sentiment, submissions_df['text'],
        desc="Submissions", chunksize=_chunksize_sub, max_workers=_n_workers,
    )
    submissions_df['sentiment_category'] = submissions_df['sentiment_score'].apply(categorize_sentiment)

    # Analyze sentiment for comments
    if not comments_df.empty:
        print("\nAnalyzing sentiment for comments...")
        _n_com = len(comments_df)
        _chunksize_com = max(1, _n_com // _n_workers)
        comments_df['sentiment_score'] = process_map(
            analyze_sentiment, comments_df['body'],
            desc="Comments", chunksize=_chunksize_com, max_workers=_n_workers,
        )
        comments_df['sentiment_category'] = comments_df['sentiment_score'].apply(categorize_sentiment)

    # Convert dates to datetime
    submissions_df['created_date'] = pd.to_datetime(submissions_df['created_date'])
    if not comments_df.empty:
        comments_df['created_date'] = pd.to_datetime(comments_df['created_date'])

    # Add category column for Autism vs ADHD
    autism_subs = ['autism', 'aspergers', 'aspergirls', 'AutisticAdults']
    autism_set = {s.lower() for s in autism_subs}
    submissions_df['category'] = submissions_df['subreddit'].apply(
        lambda x: 'Autism' if x.lower() in autism_set else 'ADHD'
    )
    if not comments_df.empty:
        comments_df['category'] = comments_df['subreddit'].apply(
            lambda x: 'Autism' if x.lower() in autism_set else 'ADHD'
        )

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

    # Load and analyze data
    submissions_df, comments_df = load_and_analyze_data()

    # Save analyzed data
    print("\nSaving analyzed data...")
    submissions_df.to_csv('reddit_submissions_with_sentiment_2026.csv', index=False)
    comments_df.to_csv('reddit_comments_with_sentiment_2026.csv', index=False)
    print("Saved analyzed data to CSV files")

    # Generate visualizations
    create_visualizations(submissions_df, comments_df)

    # Generate statistics
    stats = generate_statistics(submissions_df, comments_df)

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
