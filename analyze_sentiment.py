#!/usr/bin/env python3
"""
Script to analyze sentiment of Reddit posts and comments about Autism and ADHD.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from tqdm import tqdm

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

def load_and_analyze_data():
    """
    Load data and perform sentiment analysis.
    """
    print("Loading data...")
    submissions_df = pd.read_csv('reddit_submissions.csv')
    comments_df = pd.read_csv('reddit_comments.csv')

    # Analyze sentiment for submissions
    print("\nAnalyzing sentiment for submissions...")
    # Combine title and selftext for submissions
    submissions_df['text'] = submissions_df['title'].fillna('') + ' ' + submissions_df['selftext'].fillna('')
    tqdm.pandas(desc="Submissions")
    submissions_df['sentiment_score'] = submissions_df['text'].progress_apply(analyze_sentiment)
    submissions_df['sentiment_category'] = submissions_df['sentiment_score'].apply(categorize_sentiment)

    # Analyze sentiment for comments
    print("\nAnalyzing sentiment for comments...")
    tqdm.pandas(desc="Comments")
    comments_df['sentiment_score'] = comments_df['body'].progress_apply(analyze_sentiment)
    comments_df['sentiment_category'] = comments_df['sentiment_score'].apply(categorize_sentiment)

    # Convert dates to datetime
    submissions_df['created_date'] = pd.to_datetime(submissions_df['created_date'])
    comments_df['created_date'] = pd.to_datetime(comments_df['created_date'])

    # Add category column for Autism vs ADHD
    autism_subs = ['autism', 'aspergers', 'aspergirls', 'AutisticAdults']
    submissions_df['category'] = submissions_df['subreddit'].apply(
        lambda x: 'Autism' if x.lower() in [s.lower() for s in autism_subs] else 'ADHD'
    )
    comments_df['category'] = comments_df['subreddit'].apply(
        lambda x: 'Autism' if x.lower() in [s.lower() for s in autism_subs] else 'ADHD'
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

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Reddit Sentiment Analysis: Autism and ADHD Communities', fontsize=16, fontweight='bold')

    # 1. Sentiment distribution for submissions
    sentiment_counts = submissions_df['sentiment_category'].value_counts()
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0, 0].set_title('Submission Sentiment Distribution')

    # 2. Sentiment distribution for comments
    comment_sentiment_counts = comments_df['sentiment_category'].value_counts()
    axes[0, 1].pie(comment_sentiment_counts.values, labels=comment_sentiment_counts.index, autopct='%1.1f%%',
                   colors=['#ff9999', '#66b3ff', '#99ff99'])
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
    comments_monthly = comments_df.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
    axes[1, 1].plot(comments_monthly.index, comments_monthly.values, marker='o', linewidth=2, color='orange')
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
    category_sentiment_com = comments_df.groupby('category')['sentiment_score'].mean()
    axes[2, 1].bar(category_sentiment_com.index, category_sentiment_com.values, color=['#8B4789', '#4A90E2'])
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
    for category in ['Autism', 'ADHD']:
        data = comments_df[comments_df['category'] == category]
        monthly = data.groupby(pd.Grouper(key='created_date', freq='ME'))['sentiment_score'].mean()
        axes[1].plot(monthly.index, monthly.values, marker='o', linewidth=2, label=category)

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
    com_by_subreddit = comments_df.groupby('subreddit')['sentiment_score'].mean().sort_values()
    axes[1].barh(com_by_subreddit.index, com_by_subreddit.values, color='coral')
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
    print(f"Average sentiment score: {comments_df['sentiment_score'].mean():.4f}")
    print(f"Median sentiment score: {comments_df['sentiment_score'].median():.4f}")
    print(f"Standard deviation: {comments_df['sentiment_score'].std():.4f}")
    print("\nSentiment distribution:")
    print(comments_df['sentiment_category'].value_counts())
    print("\nBy category:")
    print(comments_df.groupby('category')['sentiment_score'].agg(['count', 'mean', 'std']))

    # Time range
    print("\n" + "-"*60)
    print("\nTIME RANGE:")
    print(f"Submissions: {submissions_df['created_date'].min()} to {submissions_df['created_date'].max()}")
    print(f"Comments: {comments_df['created_date'].min()} to {comments_df['created_date'].max()}")

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
            'avg_sentiment': comments_df['sentiment_score'].mean(),
            'median_sentiment': comments_df['sentiment_score'].median(),
            'std_sentiment': comments_df['sentiment_score'].std(),
            'sentiment_distribution': comments_df['sentiment_category'].value_counts().to_dict(),
            'by_category': comments_df.groupby('category')['sentiment_score'].mean().to_dict()
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
    submissions_df.to_csv('reddit_submissions_with_sentiment.csv', index=False)
    comments_df.to_csv('reddit_comments_with_sentiment.csv', index=False)
    print("Saved analyzed data to CSV files")

    # Generate visualizations
    create_visualizations(submissions_df, comments_df)

    # Generate statistics
    stats = generate_statistics(submissions_df, comments_df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("- reddit_submissions_with_sentiment.csv")
    print("- reddit_comments_with_sentiment.csv")
    print("- sentiment_analysis_overview.png")
    print("- sentiment_by_category.png")
    print("- sentiment_by_subreddit.png")

if __name__ == "__main__":
    main()
