"""
Sentiment analysis using VADER and TextBlob for Reddit AI discourse.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
import warnings
from tqdm import tqdm
from scipy import stats
import os
import config

warnings.filterwarnings('ignore')


def download_nltk_data():
    """Download required NLTK resources."""
    resources = ['vader_lexicon', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    punkt_resources = ['punkt_tab', 'punkt']

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")

    punkt_downloaded = False
    for punkt_resource in punkt_resources:
        try:
            nltk.download(punkt_resource, quiet=True)
            punkt_downloaded = True
            break
        except:
            continue

    if not punkt_downloaded:
        print("Warning: Punkt tokenizer unavailable. Using fallback.")


download_nltk_data()


class SentimentAnalyzer:
    """Analyze sentiment in Reddit comments using VADER and TextBlob."""

    def __init__(self, data_path):
        """
        Initialize analyzer.

        Args:
            data_path: Path to CSV file with Reddit comments
        """
        self.data_path = data_path
        self.data = None

        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def load_data(self):
        """Load and preprocess Reddit comments."""
        print("Loading Reddit comments...")

        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data):,} comments")
        print(f"Date range: {self.data['created_date'].min()} to {self.data['created_date'].max()}")

        # Standardize alignment column
        if 'community_type' in self.data.columns:
            self.data['alignment'] = self.data['community_type']

        if 'alignment' in self.data.columns:
            # Map old format to new format
            self.data['alignment'] = self.data['alignment'].replace({
                'eacc': 'e/acc',
                'ea': 'EA'
            })
        else:
            self.data['alignment'] = self.data['subreddit'].apply(self._categorize_subreddit)

        self.data['created_date'] = pd.to_datetime(self.data['created_date'])
        self.data['created_utc'] = pd.to_datetime(self.data['created_utc'], unit='s')

        # Filter date range
        start_date = pd.to_datetime('2022-01-01')
        end_date = pd.to_datetime('2024-12-31')

        initial_count = len(self.data)
        self.data = self.data[
            (self.data['created_date'] >= start_date) &
            (self.data['created_date'] <= end_date)
        ]
        print(f"Filtered to 2022-2024: {len(self.data):,} comments")
        print(f"Removed {initial_count - len(self.data):,} outside date range")

        # Remove missing text
        self.data = self.data.dropna(subset=['comment_text'])

        # Remove deleted/removed comments
        self.data = self.data[
            ~self.data['comment_text'].str.contains('removed|deleted', case=False, na=False)
        ]
        print(f"Final dataset: {len(self.data):,} comments")

        return self.data

    def _categorize_subreddit(self, subreddit):
        """
        Categorize subreddit by alignment (fallback if no alignment column).

        Args:
            subreddit: Subreddit name

        Returns:
            Alignment category: 'e/acc', 'EA', 'neutral', or 'other'
        """
        eacc_subreddits = [
            'eacc', 'Accelerationism101', 'Futurology', 'Futurism', 'AcceleratingAI',
            'ArtificialInteligence', 'OpenAI', 'LocalLLaMA', 'Automate', 'neuralnetworks',
            'SiliconValley', 'venturecapital', 'TheAllinPodcasts', 'ChatGPT'
        ]

        ea_subreddits = [
            'EffectiveAltruism', 'slatestarcodex', 'LessWrong', 'collapse',
            'ControlProblem', 'longevity', 'ClaudeAI', 'Anthropic'
        ]

        neutral_subreddits = [
            'philosophy', 'programming', 'technology', 'singularity', 'transhumanism'
        ]

        if subreddit in eacc_subreddits:
            return 'e/acc'
        elif subreddit in ea_subreddits:
            return 'EA'
        elif subreddit in neutral_subreddits:
            return 'neutral'
        else:
            return 'other'

    def preprocess_text(self, text):
        """
        Clean and preprocess text.

        Args:
            text: Raw comment text

        Returns:
            Preprocessed text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove URLs and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()

        # Remove stopwords and lemmatize
        processed = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(processed)

    def analyze_sentiment(self, sample_size=None):
        """Perform sentiment analysis using VADER and TextBlob.

        Args:
            sample_size: If provided, analyze only this many comments (for testing)
        """
        if sample_size:
            print(f"\nUsing sample of {sample_size:,} comments for testing...")
            self.data = self.data.sample(n=min(sample_size, len(self.data)), random_state=42)

        print("\nStarting sentiment analysis...")
        print(f"Analyzing {len(self.data):,} comments")

        self.data['vader_compound'] = 0.0
        self.data['vader_positive'] = 0.0
        self.data['vader_negative'] = 0.0
        self.data['vader_neutral'] = 0.0
        self.data['textblob_polarity'] = 0.0
        self.data['textblob_subjectivity'] = 0.0
        self.data['processed_text'] = ''

        tqdm.pandas(desc="Analyzing sentiment")

        def process_comment(comment):
            """Process individual comment."""
            processed = self.preprocess_text(comment)
            vader_scores = self.sia.polarity_scores(comment)
            blob = TextBlob(comment)

            return pd.Series({
                'processed_text': processed,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            })

        sentiment_results = self.data['comment_text'].progress_apply(process_comment)

        for col in sentiment_results.columns:
            self.data[col] = sentiment_results[col]

        print("Sentiment analysis complete")
        self._print_summary()

    def _print_summary(self):
        """Print sentiment analysis summary."""
        print("\n=== Sentiment Analysis Summary ===")
        print(f"VADER Compound: {self.data['vader_compound'].min():.3f} to {self.data['vader_compound'].max():.3f}")
        print(f"VADER Mean: {self.data['vader_compound'].mean():.3f}")
        print(f"TextBlob Polarity: {self.data['textblob_polarity'].min():.3f} to {self.data['textblob_polarity'].max():.3f}")
        print(f"TextBlob Mean: {self.data['textblob_polarity'].mean():.3f}")

        correlation = self.data['vader_compound'].corr(self.data['textblob_polarity'])
        print(f"VADER-TextBlob Correlation: {correlation:.3f}")

    def analyze_by_alignment(self):
        """Analyze sentiment differences by ideological alignment."""
        print("\n=== Analyzing by alignment ===")

        alignment_data = self.data[self.data['alignment'].isin(['e/acc', 'EA', 'neutral'])]

        stats_by_alignment = alignment_data.groupby('alignment').agg({
            'vader_compound': ['mean', 'std', 'median', 'count'],
            'textblob_polarity': ['mean', 'std', 'median', 'count']
        }).round(4)

        print("\nSentiment by alignment:")
        print(stats_by_alignment)

        # ANOVA
        eacc = alignment_data[alignment_data['alignment'] == 'e/acc']['vader_compound']
        ea = alignment_data[alignment_data['alignment'] == 'EA']['vader_compound']
        neutral = alignment_data[alignment_data['alignment'] == 'neutral']['vader_compound']

        f_stat, p_value = stats.f_oneway(eacc, ea, neutral)
        print(f"\nANOVA: F={f_stat:.4f}, p={p_value:.4e}")

        # Pairwise t-tests
        from scipy.stats import ttest_ind

        print("\nPairwise comparisons:")
        t1, p1 = ttest_ind(eacc, ea)
        print(f"e/acc vs EA: t={t1:.4f}, p={p1:.4e}")

        t2, p2 = ttest_ind(eacc, neutral)
        print(f"e/acc vs neutral: t={t2:.4f}, p={p2:.4e}")

        t3, p3 = ttest_ind(ea, neutral)
        print(f"EA vs neutral: t={t3:.4f}, p={p3:.4e}")

        return {
            'stats': stats_by_alignment,
            'anova': {'f_stat': f_stat, 'p_value': p_value},
            'pairwise': {
                'eacc_vs_ea': {'t': t1, 'p': p1},
                'eacc_vs_neutral': {'t': t2, 'p': p2},
                'ea_vs_neutral': {'t': t3, 'p': p3}
            }
        }

    def save_results(self):
        """Save sentiment analysis results to CSV."""
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

        # Match exact column order from thesis output
        output_columns = [
            'comment_id', 'subreddit', 'comment_text', 'created_date', 'created_utc',
            'alignment', 'vader_compound', 'vader_positive', 'vader_negative',
            'vader_neutral', 'textblob_polarity', 'textblob_subjectivity',
            'processed_text', 'analysis_date', 'sample_size'
        ]

        export_data = self.data.copy()

        # Add metadata columns
        export_data['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        export_data['sample_size'] = len(self.data)

        # Ensure created_utc exists (convert from created_date if needed)
        if 'created_utc' not in export_data.columns:
            export_data['created_utc'] = pd.to_datetime(export_data['created_date']).astype(int) / 10**9

        # Select only the columns we need in the correct order
        export_data = export_data[output_columns]

        filename = os.path.join(
            config.PROCESSED_DATA_DIR,
            f'reddit_sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

        export_data.to_csv(filename, index=False)

        print(f"\nSaved {len(export_data):,} comments to {filename}")

        return filename


def main(sample_size=None):
    """
    Run sentiment analysis pipeline.

    Args:
        sample_size: If provided, analyze only this many comments (for testing)
    """
    analyzer = SentimentAnalyzer(config.RAW_COMMENTS_FILE)

    analyzer.load_data()
    analyzer.analyze_sentiment(sample_size=sample_size)
    alignment_results = analyzer.analyze_by_alignment()
    output_file = analyzer.save_results()

    print("\n=== Analysis Complete ===")
    print(f"Output: {output_file}")
    print("Use R scripts in analysis/ for visualizations and statistical tests")

    return analyzer, alignment_results


if __name__ == "__main__":
    # Test with 1000 comments: main(sample_size=1000)
    # Full analysis: main()
    import sys

    if len(sys.argv) > 1:
        sample = int(sys.argv[1])
        print(f"Running with sample size: {sample:,}")
        main(sample_size=sample)
    else:
        main()