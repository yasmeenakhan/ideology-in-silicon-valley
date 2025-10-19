"""
Configuration file for Reddit scraping and sentiment analysis.
"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Reddit API Credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'IdeologyAnalysis/1.0')

# Date range for data collection
START_DATE = datetime(2022, 1, 1).timestamp()
END_DATE = datetime(2024, 12, 31).timestamp()

# Subreddit categorizations
EACC_SUBREDDITS = [
    'eacc', 'Accelerationism101', 'Futurology', 'Futurism', 'AcceleratingAI',
    'ArtificialInteligence', 'OpenAI', 'LocalLLaMA', 'Automate', 'neuralnetworks',
    'SiliconValley', 'venturecapital', 'TheAllinPodcasts', 'ChatGPT'
]

EA_SUBREDDITS = [
    'EffectiveAltruism', 'slatestarcodex', 'LessWrong', 'collapse',
    'ControlProblem', 'longevity', 'ClaudeAI', 'Anthropic'
]

NEUTRAL_SUBREDDITS = [
    'philosophy', 'programming', 'technology', 'singularity', 'transhumanism'
]

# Scraping parameters
POSTS_LIMIT = 1000

# File paths
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

RAW_POSTS_FILE = os.path.join(RAW_DATA_DIR, 'reddit_posts.csv')
RAW_COMMENTS_FILE = os.path.join(RAW_DATA_DIR, 'reddit_comments.csv')

# Visualization configuration
VIZ_CONFIG = {
    'alignment_order': ['e/acc', 'neutral', 'EA'],
    'colors': {
        'e/acc': '#CA6702',
        'neutral': '#94A187',
        'EA': '#0A9396'
    }
}

# Key events for temporal analysis
KEY_EVENTS = {
    'eacc_launch': '2022-06-01',
    'chatgpt_launch': '2022-11-30',
    'andreessen_manifesto': '2023-10-16'
}

# Major technology subreddits (>1M subscribers)
MAJOR_TECH_SUBREDDITS = [
    'technology', 'ChatGPT', 'singularity', 'ArtificialInteligence',
    'OpenAI', 'programming', 'Futurology'
]