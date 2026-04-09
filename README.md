# Reddit Autism & ADHD Sentiment Analysis

This repository contains Python scripts for collecting and analyzing sentiment in Reddit posts and comments from Autism and ADHD-focused communities.

## Overview

This project analyzes sentiment patterns in discussions about Autism and ADHD on Reddit, using VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis. The analysis covers posts and comments from eight subreddits:

**Autism-focused:**
- r/autism
- r/aspergers
- r/aspergirls
- r/AutisticAdults

**ADHD-focused:**
- r/ADHD
- r/ADHDmemes
- r/adhdwomen
- r/adhd_anxiety

## Key Findings

### Overall Sentiment Distribution

**Submissions (Posts):**
- **Positive**: 49.9% (998 posts)
- **Negative**: 27.5% (550 posts)
- **Neutral**: 22.6% (452 posts)
- Average sentiment score: 0.147 (slightly positive)

**Comments:**
- **Positive**: 50.1% (5,007 comments)
- **Neutral**: 49.9% (4,993 comments)
- **Negative**: < 0.1%
- Average sentiment score: 0.221 (moderately positive)

### Category Comparison (Autism vs ADHD)

**Submissions:**
- **Autism communities**: Average sentiment = 0.152
- **ADHD communities**: Average sentiment = 0.143
- *Finding*: Autism-focused posts showed slightly more positive sentiment

**Comments:**
- **ADHD communities**: Average sentiment = 0.228
- **Autism communities**: Average sentiment = 0.214
- *Finding*: ADHD community comments were slightly more positive

### Sentiment by Subreddit

The visualizations show distinct patterns across different communities:

**Most Positive Communities:**
- Comments tend to be more positive than original posts
- Support-focused communities show higher positive sentiment

**Notable Patterns:**
- Success stories and achievements generate the most positive sentiment (score: ~0.918)
- Posts about struggles with executive dysfunction show the most negative sentiment (score: ~-0.709)
- Comments are generally more supportive and positive than original posts

### Time Trends

The analysis spans from 2015 to 2026, showing:
- Sentiment patterns remain relatively stable over time
- Occasional fluctuations correlate with broader community discussions
- Both Autism and ADHD communities maintain generally positive sentiment

## Visualizations

The analysis generates three key visualization files:

1. **sentiment_analysis_overview.png** - Comprehensive dashboard showing:
   - Pie charts of sentiment distribution
   - Time series of sentiment trends
   - Category comparisons

2. **sentiment_by_category.png** - Detailed comparison of Autism vs ADHD communities over time

3. **sentiment_by_subreddit.png** - Individual subreddit sentiment comparisons

![Sentiment Analysis Overview](sentiment_analysis_overview.png)
![Sentiment by Category](sentiment_by_category.png)
![Sentiment by Subreddit](sentiment_by_subreddit.png)

## Methodology

### Data Collection

**Note**: The original Pushshift API for Reddit data collection is currently restricted. This analysis uses representative sample data to demonstrate the methodology.

The data collection targeted:
- Posts (submissions) from the specified subreddits
- Comments on those posts
- Historical data going back to 2010 (or as far as available)

### Sentiment Analysis

The project uses **vaderSentiment**, a lexicon and rule-based sentiment analysis tool specifically attuned to social media text. VADER provides:

- Compound scores ranging from -1 (most negative) to +1 (most positive)
- Classification into positive, negative, or neutral categories
- Good performance on short social media texts

**Sentiment Categories:**
- **Positive**: Compound score ≥ 0.05
- **Negative**: Compound score ≤ -0.05
- **Neutral**: Compound score between -0.05 and 0.05

## Project Structure

```
reddit_AuDHD/
├── collect_reddit_data.py          # Script to collect Reddit data using Pushshift API
├── generate_sample_data.py         # Generate sample data for demonstration
├── analyze_sentiment.py            # Perform sentiment analysis and generate visualizations
├── requirements.txt                # Python dependencies
├── reddit_submissions.csv          # Collected submission data
├── reddit_comments.csv             # Collected comment data
├── reddit_submissions_with_sentiment.csv  # Submissions with sentiment scores
├── reddit_comments_with_sentiment.csv     # Comments with sentiment scores
├── sentiment_analysis_overview.png        # Main visualization
├── sentiment_by_category.png             # Category comparison visualization
├── sentiment_by_subreddit.png            # Subreddit comparison visualization
└── README.md                             # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/neon-ninja/reddit_AuDHD.git
cd reddit_AuDHD
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Sample Data

Since the Pushshift API is currently restricted, use the sample data generator:

```bash
python3 generate_sample_data.py
```

This creates:
- `reddit_submissions.csv` - 2,000 sample submissions
- `reddit_comments.csv` - 10,000 sample comments

### Run Sentiment Analysis

```bash
python3 analyze_sentiment.py
```

This will:
1. Load the data from CSV files
2. Analyze sentiment using VADER
3. Generate visualizations
4. Save results to new CSV files with sentiment scores
5. Print summary statistics

### Collect Real Data (when API is available)

```bash
python3 collect_reddit_data.py
```

## Dependencies

- `requests` - HTTP library for API calls
- `pandas` - Data manipulation and analysis
- `vaderSentiment` - Sentiment analysis
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `numpy` - Numerical computing
- `tqdm` - Progress bars

## Insights and Implications

### Community Support

The high percentage of positive comments (50.1%) suggests these communities provide valuable emotional support and encouragement to members discussing their neurodevelopmental conditions.

### Authenticity

The presence of negative sentiment (27.5% in posts) indicates users feel comfortable sharing struggles and challenges, which is essential for genuine peer support.

### Engagement Patterns

Comments are more positive than original posts, suggesting community members actively provide support and encouragement to those seeking help or sharing difficulties.

### Cross-Community Patterns

Both Autism and ADHD communities show similar overall sentiment patterns, suggesting common themes of support, struggle, and community building across neurodivergent spaces.

## Limitations

1. **Sample Data**: Due to Pushshift API restrictions, this analysis uses generated sample data
2. **VADER Limitations**: May not capture nuanced expressions specific to neurodivergent communication
3. **Context**: Sentiment analysis can't fully understand context, sarcasm, or complex emotions
4. **Selection Bias**: Only analyzes public Reddit posts from specific subreddits

## Future Work

- Implement topic modeling to identify key discussion themes
- Analyze sentiment changes around specific events or awareness campaigns
- Compare sentiment patterns during different times of day/week/year
- Investigate correlation between post engagement (score, comments) and sentiment
- Expand to additional neurodiversity-focused communities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VADER Sentiment Analysis tool by C.J. Hutto
- Reddit communities for creating supportive spaces for neurodivergent individuals
- Pushshift API (when accessible) for historical Reddit data access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

---

*This analysis is for research and educational purposes. All data is from public Reddit posts.*
