#!/usr/bin/env python3
"""
Script to collect Reddit posts and comments about Autism and ADHD using the Pushshift API.
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
from tqdm import tqdm

# Subreddits focused on Autism and ADHD
AUTISM_SUBREDDITS = ['autism', 'aspergers', 'aspergirls', 'AutisticAdults']
ADHD_SUBREDDITS = ['ADHD', 'ADHDmemes', 'adhdwomen', 'adhd_anxiety']
ALL_SUBREDDITS = AUTISM_SUBREDDITS + ADHD_SUBREDDITS

# Pushshift API endpoints
PUSHSHIFT_SUBMISSION_URL = "https://api.pushshift.io/reddit/search/submission/"
PUSHSHIFT_COMMENT_URL = "https://api.pushshift.io/reddit/search/comment/"

# Start from 2010 (Reddit's early days, pushshift data availability)
START_TIMESTAMP = int(datetime(2010, 1, 1).timestamp())
END_TIMESTAMP = int(datetime.now().timestamp())

def fetch_submissions(subreddit, start_time, end_time, limit=100):
    """
    Fetch submissions from a subreddit within a time range.
    """
    params = {
        'subreddit': subreddit,
        'after': start_time,
        'before': end_time,
        'size': limit,
        'sort': 'created_utc',
        'sort_type': 'asc'
    }

    try:
        response = requests.get(PUSHSHIFT_SUBMISSION_URL, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            print(f"Error {response.status_code} fetching submissions from r/{subreddit}")
            return []
    except Exception as e:
        print(f"Exception fetching submissions: {e}")
        return []

def fetch_comments(subreddit, start_time, end_time, limit=100):
    """
    Fetch comments from a subreddit within a time range.
    """
    params = {
        'subreddit': subreddit,
        'after': start_time,
        'before': end_time,
        'size': limit,
        'sort': 'created_utc',
        'sort_type': 'asc'
    }

    try:
        response = requests.get(PUSHSHIFT_COMMENT_URL, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            print(f"Error {response.status_code} fetching comments from r/{subreddit}")
            return []
    except Exception as e:
        print(f"Exception fetching comments: {e}")
        return []

def collect_data_for_subreddit(subreddit, start_time, end_time, data_type='submissions'):
    """
    Collect all data for a subreddit by paginating through time windows.
    """
    all_data = []
    current_time = start_time
    batch_size = 500  # Maximum allowed by Pushshift

    print(f"\nCollecting {data_type} from r/{subreddit}...")

    fetch_function = fetch_submissions if data_type == 'submissions' else fetch_comments

    with tqdm(total=end_time - start_time, desc=f"r/{subreddit}") as pbar:
        while current_time < end_time:
            batch = fetch_function(subreddit, current_time, end_time, limit=batch_size)

            if not batch:
                # If no data returned, jump forward in time
                current_time += 86400 * 30  # 30 days
                pbar.update(86400 * 30)
                time.sleep(0.5)  # Rate limiting
                continue

            all_data.extend(batch)

            # Update current_time to the timestamp of the last item
            last_timestamp = batch[-1].get('created_utc', current_time)
            time_jump = last_timestamp - current_time + 1
            current_time = last_timestamp + 1
            pbar.update(time_jump)

            # Rate limiting - be nice to the API
            time.sleep(0.5)

            # Stop if we got fewer items than requested (reached the end)
            if len(batch) < batch_size:
                break

    print(f"Collected {len(all_data)} {data_type} from r/{subreddit}")
    return all_data

def extract_submission_features(submissions):
    """
    Extract relevant features from submissions.
    """
    extracted = []
    for sub in submissions:
        extracted.append({
            'id': sub.get('id'),
            'subreddit': sub.get('subreddit'),
            'title': sub.get('title', ''),
            'selftext': sub.get('selftext', ''),
            'author': sub.get('author'),
            'score': sub.get('score', 0),
            'num_comments': sub.get('num_comments', 0),
            'created_utc': sub.get('created_utc'),
            'created_date': datetime.fromtimestamp(sub.get('created_utc', 0)).strftime('%Y-%m-%d'),
            'url': sub.get('url', ''),
            'is_self': sub.get('is_self', False)
        })
    return extracted

def extract_comment_features(comments):
    """
    Extract relevant features from comments.
    """
    extracted = []
    for comment in comments:
        extracted.append({
            'id': comment.get('id'),
            'subreddit': comment.get('subreddit'),
            'body': comment.get('body', ''),
            'author': comment.get('author'),
            'score': comment.get('score', 0),
            'created_utc': comment.get('created_utc'),
            'created_date': datetime.fromtimestamp(comment.get('created_utc', 0)).strftime('%Y-%m-%d'),
            'parent_id': comment.get('parent_id', ''),
            'link_id': comment.get('link_id', '')
        })
    return extracted

def main():
    """
    Main function to orchestrate data collection.
    """
    print("Starting Reddit data collection for Autism and ADHD subreddits")
    print(f"Target subreddits: {', '.join(ALL_SUBREDDITS)}")
    print(f"Time range: {datetime.fromtimestamp(START_TIMESTAMP)} to {datetime.fromtimestamp(END_TIMESTAMP)}")

    all_submissions = []
    all_comments = []

    # Collect submissions from all subreddits
    print("\n" + "="*60)
    print("COLLECTING SUBMISSIONS")
    print("="*60)
    for subreddit in ALL_SUBREDDITS:
        submissions = collect_data_for_subreddit(subreddit, START_TIMESTAMP, END_TIMESTAMP, 'submissions')
        all_submissions.extend(extract_submission_features(submissions))

    # Collect comments from all subreddits
    print("\n" + "="*60)
    print("COLLECTING COMMENTS")
    print("="*60)
    for subreddit in ALL_SUBREDDITS:
        comments = collect_data_for_subreddit(subreddit, START_TIMESTAMP, END_TIMESTAMP, 'comments')
        all_comments.extend(extract_comment_features(comments))

    # Convert to DataFrames
    submissions_df = pd.DataFrame(all_submissions)
    comments_df = pd.DataFrame(all_comments)

    # Save to CSV
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    submissions_df.to_csv('reddit_submissions.csv', index=False)
    comments_df.to_csv('reddit_comments.csv', index=False)

    print(f"\nTotal submissions collected: {len(submissions_df)}")
    print(f"Total comments collected: {len(comments_df)}")
    print(f"\nData saved to reddit_submissions.csv and reddit_comments.csv")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("\nSubmissions by subreddit:")
    print(submissions_df['subreddit'].value_counts())
    print("\nComments by subreddit:")
    print(comments_df['subreddit'].value_counts())

if __name__ == "__main__":
    main()
