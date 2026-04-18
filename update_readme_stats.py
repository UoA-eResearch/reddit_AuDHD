#!/usr/bin/env python3
"""
Script to update README.md with current statistics from the analyzed data.
"""

import pandas as pd
import re
from datetime import datetime

CHUNK_SIZE = 50_000
SUBMISSIONS_FILE = 'reddit_submissions_with_sentiment_2026.csv'
USECOLS = ['author_hash', 'subreddit', 'created_date', 'sentiment_category',
           'sentiment_score', 'category', 'title']
AUTISM_SUBS = {'autism', 'aspergers', 'aspergirls', 'autisticadults'}


def calculate_statistics():
    """Calculate all statistics needed for the README by streaming the CSV in chunks."""

    total_posts = 0
    author_hashes: set = set()
    autism_posts = 0
    date_min = None
    date_max = None
    sentiment_cat_counts: dict = {}
    sentiment_sum = 0.0
    category_sentiment_sum: dict = {}
    category_sentiment_count: dict = {}
    subreddit_sentiment_sum: dict = {}
    subreddit_sentiment_count: dict = {}
    most_positive_score = None
    most_positive_row = None
    most_negative_score = None
    most_negative_row = None

    for chunk in pd.read_csv(SUBMISSIONS_FILE, chunksize=CHUNK_SIZE, usecols=USECOLS):
        chunk['created_date'] = pd.to_datetime(chunk['created_date'])

        total_posts += len(chunk)
        author_hashes.update(chunk['author_hash'].dropna().unique())
        autism_posts += int(chunk['subreddit'].str.lower().isin(AUTISM_SUBS).sum())

        chunk_min = chunk['created_date'].min()
        chunk_max = chunk['created_date'].max()
        if date_min is None or chunk_min < date_min:
            date_min = chunk_min
        if date_max is None or chunk_max > date_max:
            date_max = chunk_max

        for cat, cnt in chunk['sentiment_category'].value_counts().items():
            sentiment_cat_counts[cat] = sentiment_cat_counts.get(cat, 0) + int(cnt)

        sentiment_sum += float(chunk['sentiment_score'].sum())

        for cat, grp in chunk.groupby('category')['sentiment_score']:
            category_sentiment_sum[cat] = category_sentiment_sum.get(cat, 0.0) + float(grp.sum())
            category_sentiment_count[cat] = category_sentiment_count.get(cat, 0) + len(grp)

        for sub, grp in chunk.groupby('subreddit')['sentiment_score']:
            subreddit_sentiment_sum[sub] = subreddit_sentiment_sum.get(sub, 0.0) + float(grp.sum())
            subreddit_sentiment_count[sub] = subreddit_sentiment_count.get(sub, 0) + len(grp)

        valid_scores = chunk['sentiment_score'].dropna()
        if not valid_scores.empty:
            chunk_max_idx = valid_scores.idxmax()
            chunk_min_idx = valid_scores.idxmin()
            chunk_max_score = float(chunk.at[chunk_max_idx, 'sentiment_score'])
            chunk_min_score = float(chunk.at[chunk_min_idx, 'sentiment_score'])
            if most_positive_score is None or chunk_max_score > most_positive_score:
                most_positive_score = chunk_max_score
                most_positive_row = chunk.loc[chunk_max_idx]
            if most_negative_score is None or chunk_min_score < most_negative_score:
                most_negative_score = chunk_min_score
                most_negative_row = chunk.loc[chunk_min_idx]

    unique_authors = len(author_hashes)
    adhd_posts = total_posts - autism_posts
    avg_sentiment = sentiment_sum / total_posts if total_posts > 0 else 0.0

    autism_sentiment = (category_sentiment_sum.get('Autism', 0.0) /
                        category_sentiment_count['Autism']
                        if 'Autism' in category_sentiment_count else 0.0)
    adhd_sentiment = (category_sentiment_sum.get('ADHD', 0.0) /
                      category_sentiment_count['ADHD']
                      if 'ADHD' in category_sentiment_count else 0.0)

    subreddit_sentiment = pd.Series({
        sub: subreddit_sentiment_sum[sub] / subreddit_sentiment_count[sub]
        for sub in subreddit_sentiment_sum
    }).sort_values(ascending=False)

    positive_count = sentiment_cat_counts.get('positive', 0)
    negative_count = sentiment_cat_counts.get('negative', 0)
    neutral_count = sentiment_cat_counts.get('neutral', 0)
    positive_pct = positive_count / total_posts * 100 if total_posts > 0 else 0
    negative_pct = negative_count / total_posts * 100 if total_posts > 0 else 0
    neutral_pct = neutral_count / total_posts * 100 if total_posts > 0 else 0

    return {
        'total_posts': total_posts,
        'unique_authors': unique_authors,
        'autism_posts': autism_posts,
        'adhd_posts': adhd_posts,
        'date_min': date_min.strftime('%Y-%m-%d') if date_min is not None else 'N/A',
        'date_max': date_max.strftime('%Y-%m-%d') if date_max is not None else 'N/A',
        'positive_count': positive_count,
        'positive_pct': positive_pct,
        'negative_count': negative_count,
        'negative_pct': negative_pct,
        'neutral_count': neutral_count,
        'neutral_pct': neutral_pct,
        'avg_sentiment': avg_sentiment,
        'autism_sentiment': autism_sentiment,
        'adhd_sentiment': adhd_sentiment,
        'subreddit_sentiment': subreddit_sentiment,
        'most_positive': most_positive_row,
        'most_negative': most_negative_row,
    }


def update_readme(stats):
    """Update README.md with new statistics."""

    with open('README.md', 'r') as f:
        readme = f.read()

    # Update the "Initial Results" section heading to show it's auto-updated
    readme = re.sub(
        r'## Initial Results \(Seed Collection — April 2026\)',
        f'## Dataset Statistics (Last Updated: {datetime.now().strftime("%Y-%m-%d")})',
        readme
    )

    # Update the note about seed collection
    readme = re.sub(
        r'A seed dataset was collected on 2026-04-09.*?dataset over time\.',
        f'Data collection began on 2026-04-09 via Tor (with automatic exit-node rotation). The weekly GitHub Actions workflow continues to grow this dataset over time.',
        readme,
        flags=re.DOTALL
    )

    # Update total posts
    readme = re.sub(
        r'\| Total posts \| \*\*[0-9,]+\*\* \|',
        f'| Total posts | **{stats["total_posts"]:,}** |',
        readme
    )

    # Update unique redditors
    readme = re.sub(
        r'\| Unique redditors \(posts\) \| \*\*[0-9,]+\*\*',
        f'| Unique redditors (posts) | **{stats["unique_authors"]:,}**',
        readme
    )

    # Update autism community posts
    readme = re.sub(
        r'\| Autism-community posts \| [0-9,]+',
        f'| Autism-community posts | {stats["autism_posts"]:,}',
        readme
    )

    # Update ADHD community posts
    readme = re.sub(
        r'\| ADHD-community posts \| [0-9,]+',
        f'| ADHD-community posts | {stats["adhd_posts"]:,}',
        readme
    )

    # Update date range
    readme = re.sub(
        r'\| Date range \| [0-9]{4}-[0-9]{2}-[0-9]{2} → [0-9]{4}-[0-9]{2}-[0-9]{2} \|',
        f'| Date range | {stats["date_min"]} → {stats["date_max"]} |',
        readme
    )

    # Update sentiment distribution
    readme = re.sub(
        r'\| Positive \(score ≥ 0\.05\) \| [0-9,]+ \| [0-9.]+% \|',
        f'| Positive (score ≥ 0.05) | {stats["positive_count"]:,} | {stats["positive_pct"]:.1f}% |',
        readme
    )
    readme = re.sub(
        r'\| Negative \(score ≤ −0\.05\) \| [0-9,]+ \| [0-9.]+% \|',
        f'| Negative (score ≤ −0.05) | {stats["negative_count"]:,} | {stats["negative_pct"]:.1f}% |',
        readme
    )
    readme = re.sub(
        r'\| Neutral \| [0-9,]+ \| [0-9.]+% \|',
        f'| Neutral | {stats["neutral_count"]:,} | {stats["neutral_pct"]:.1f}% |',
        readme
    )

    # Update average compound score
    readme = re.sub(
        r'Average compound score: \*\*[0-9.-]+\*\*',
        f'Average compound score: **{stats["avg_sentiment"]:.3f}**',
        readme
    )

    # Update sentiment by community
    readme = re.sub(
        r'\| Autism \| [0-9.-]+ \|',
        f'| Autism | {stats["autism_sentiment"]:.3f} |',
        readme
    )
    readme = re.sub(
        r'\| ADHD \| [0-9.-]+ \|',
        f'| ADHD | {stats["adhd_sentiment"]:.3f} |',
        readme
    )

    # Update sentiment by subreddit - rebuild the entire table
    subreddit_table_lines = []
    for subreddit, sentiment in stats['subreddit_sentiment'].items():
        sign = '+' if sentiment >= 0 else ''
        note = ''
        if sentiment == stats['subreddit_sentiment'].max():
            note = ' (most positive)'
        elif sentiment == stats['subreddit_sentiment'].min():
            note = ' (least positive)' if sentiment > 0 else ' (only subreddit with net-negative avg.)'
        subreddit_table_lines.append(f'| r/{subreddit} | {sign}{sentiment:.3f}{note} |')

    subreddit_table = '\n'.join(subreddit_table_lines)

    # Replace the subreddit sentiment table
    readme = re.sub(
        r'(\| Subreddit \| Avg\. Sentiment \|\n\|---\|---\|\n)(\| r/.*\n)+',
        r'\1' + subreddit_table + '\n',
        readme
    )

    # Update most positive post
    most_pos = stats['most_positive']
    readme = re.sub(
        r'\*\*Most positive post:\*\* \*".*?"\*\n\(r/.*?, sentiment [0-9.-]+\)',
        f'**Most positive post:** *"{most_pos["title"][:80]}..."*\n(r/{most_pos["subreddit"]}, sentiment {most_pos["sentiment_score"]:.4f})',
        readme
    )

    # Update most negative post
    most_neg = stats['most_negative']
    readme = re.sub(
        r'\*\*Most negative post:\*\* \*".*?"\*\n\(r/.*?, sentiment [0-9.-]+\)',
        f'**Most negative post:** *"{most_neg["title"][:80]}..."*\n(r/{most_neg["subreddit"]}, sentiment {most_neg["sentiment_score"]:.4f})',
        readme
    )

    # Update the note at the bottom of results section
    readme = re.sub(
        r'> \*\*Note:\*\*.*?via the GitHub Actions cron workflow\.',
        f'> **Note:** This dataset is automatically updated weekly via GitHub Actions. Statistics shown reflect the most recent analysis run.',
        readme,
        flags=re.DOTALL
    )

    with open('README.md', 'w') as f:
        f.write(readme)

    print("README.md updated successfully!")


def main():
    """Main function."""
    print("Calculating statistics (streaming CSV in chunks)...")
    stats = calculate_statistics()

    print("Updating README.md...")
    update_readme(stats)

    print("\nStatistics updated:")
    print(f"  Total posts: {stats['total_posts']:,}")
    print(f"  Unique authors: {stats['unique_authors']:,}")
    print(f"  Date range: {stats['date_min']} → {stats['date_max']}")
    print(f"  Average sentiment: {stats['avg_sentiment']:.3f}")


if __name__ == "__main__":
    main()
