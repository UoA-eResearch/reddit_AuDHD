#!/usr/bin/env python3
"""
Script to generate sample Reddit data for demonstration purposes.
Since the Pushshift API is currently restricted, this creates realistic sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Subreddits
AUTISM_SUBREDDITS = ['autism', 'aspergers', 'aspergirls', 'AutisticAdults']
ADHD_SUBREDDITS = ['ADHD', 'ADHDmemes', 'adhdwomen', 'adhd_anxiety']
ALL_SUBREDDITS = AUTISM_SUBREDDITS + ADHD_SUBREDDITS

# Sample titles and text snippets for posts
AUTISM_TITLES = [
    "Just got diagnosed at 35, feeling relieved",
    "Does anyone else struggle with sensory overload?",
    "My experience with ASD diagnosis as an adult",
    "Looking for advice on managing meltdowns",
    "Finally found a therapist who understands autism",
    "Anyone else hyperfocused on their special interests?",
    "Struggling with social interactions at work",
    "Success story: landed my dream job!",
    "Does anyone else find eye contact exhausting?",
    "My autistic child just said their first sentence!",
]

ADHD_TITLES = [
    "Started medication today, feeling hopeful",
    "Does anyone else forget what they're saying mid-sentence?",
    "Just got diagnosed with ADHD at 28",
    "Looking for tips on time management",
    "Finally finished a project I started months ago!",
    "Does caffeine help anyone else focus?",
    "Struggling with executive dysfunction",
    "Success: I cleaned my entire apartment!",
    "Anyone else have rejection sensitive dysphoria?",
    "Tips for managing ADHD without medication?",
]

COMMENT_TEXTS = [
    "I completely relate to this!",
    "Thanks for sharing, this helps me feel less alone.",
    "I've been through something similar.",
    "Have you tried talking to a therapist about this?",
    "This is exactly how I feel!",
    "You're not alone in this journey.",
    "Proud of you for sharing!",
    "I struggled with this for years before finding help.",
    "This resonates with me so much.",
    "Thank you for posting this!",
]

def generate_submissions(num_submissions=1000):
    """Generate sample submission data"""
    submissions = []

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 4, 1)

    for i in range(num_submissions):
        subreddit = random.choice(ALL_SUBREDDITS)

        # Choose title based on subreddit category
        if subreddit in AUTISM_SUBREDDITS:
            title = random.choice(AUTISM_TITLES)
        else:
            title = random.choice(ADHD_TITLES)

        # Generate random date between start and end
        time_delta = end_date - start_date
        random_days = random.randint(0, time_delta.days)
        created_date = start_date + timedelta(days=random_days)

        submissions.append({
            'id': f'sub_{i}',
            'subreddit': subreddit,
            'title': title,
            'selftext': random.choice(COMMENT_TEXTS) if random.random() > 0.3 else '',
            'author': f'user_{random.randint(1, 200)}',
            'score': max(1, int(np.random.exponential(20))),
            'num_comments': max(0, int(np.random.exponential(15))),
            'created_utc': int(created_date.timestamp()),
            'created_date': created_date.strftime('%Y-%m-%d'),
            'url': f'https://reddit.com/r/{subreddit}/comments/{i}',
            'is_self': True
        })

    return pd.DataFrame(submissions)

def generate_comments(num_comments=5000):
    """Generate sample comment data"""
    comments = []

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2026, 4, 1)

    for i in range(num_comments):
        subreddit = random.choice(ALL_SUBREDDITS)

        # Generate random date
        time_delta = end_date - start_date
        random_days = random.randint(0, time_delta.days)
        created_date = start_date + timedelta(days=random_days)

        comments.append({
            'id': f'com_{i}',
            'subreddit': subreddit,
            'body': random.choice(COMMENT_TEXTS),
            'author': f'user_{random.randint(1, 200)}',
            'score': max(1, int(np.random.exponential(10))),
            'created_utc': int(created_date.timestamp()),
            'created_date': created_date.strftime('%Y-%m-%d'),
            'parent_id': f't3_sub_{random.randint(1, 1000)}',
            'link_id': f't3_sub_{random.randint(1, 1000)}'
        })

    return pd.DataFrame(comments)

def main():
    """Generate and save sample data"""
    print("Generating sample Reddit data for demonstration...")

    print("\nGenerating submissions...")
    submissions_df = generate_submissions(num_submissions=2000)
    print(f"Generated {len(submissions_df)} submissions")

    print("\nGenerating comments...")
    comments_df = generate_comments(num_comments=10000)
    print(f"Generated {len(comments_df)} comments")

    # Save to CSV
    print("\nSaving data to CSV files...")
    submissions_df.to_csv('reddit_submissions.csv', index=False)
    comments_df.to_csv('reddit_comments.csv', index=False)

    print("\nData generation complete!")
    print(f"Submissions saved to: reddit_submissions.csv")
    print(f"Comments saved to: reddit_comments.csv")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nSubmissions by subreddit:")
    print(submissions_df['subreddit'].value_counts())
    print("\nComments by subreddit:")
    print(comments_df['subreddit'].value_counts())

    print("\nNote: This is sample data generated for demonstration purposes.")
    print("The Pushshift API is currently restricted and unavailable for public use.")

if __name__ == "__main__":
    main()
