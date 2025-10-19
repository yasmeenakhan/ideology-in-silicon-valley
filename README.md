# Ideology in Silicon Valley

Sentiment analysis of AI discourse on Reddit, comparing effective accelerationism (e/acc) and effective altruism (EA) communities.

## Project Structure

```
ideology-in-silicon-valley/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.py
├── data/
│   ├── raw/                    # Raw Reddit data (gitignored)
│   └── processed/              # Sentiment-analyzed data
├── src/
│   ├── scraper.py             # Reddit scraping functionality
│   └── sentiment_analyzer.py  # VADER + TextBlob sentiment analysis
├── analysis/
│   ├── statistical_tests.Rmd  # Statistical tests in R
│   └── visualizations.Rmd     # Visualizations in R
└── figures/                    # Generated plots
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ideology-in-silicon-valley.git
cd ideology-in-silicon-valley
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install R dependencies

Open R/RStudio and run:

```r
install.packages(c("tidyverse", "lubridate", "showtext", "rstatix", 
                   "lsr", "effsize", "psych", "viridis"))
```

### 4. Configure Reddit API credentials

Copy `.env.example` to `.env` and add your Reddit API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your credentials from [Reddit Apps](https://www.reddit.com/prefs/apps):

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=IdeologyAnalysis/1.0
```

### 5. Add your dataset

Place your Reddit comments CSV in `data/raw/reddit_comments.csv`

The CSV should have these columns:
- `comment_id`
- `comment_text`
- `subreddit`
- `created_date`
- `alignment` (e/acc, EA, neutral, or other)
- `author`
- `score`

## Usage

### Option 1: Scrape new data

```bash
python src/scraper.py
```

This will scrape Reddit data from configured subreddits and save to `data/raw/`.

### Option 2: Analyze existing data

```bash
python src/sentiment_analyzer.py
```

This performs VADER and TextBlob sentiment analysis on `data/raw/reddit_comments.csv` and saves results to `data/processed/`.

### Generate visualizations and statistics

Open the R Markdown files in RStudio:

1. `analysis/statistical_tests.Rmd` - Run statistical tests
2. `analysis/visualizations.Rmd` - Generate publication-quality plots

Plots are saved to `figures/`.

## Data Description

### Subreddit Categories

**e/acc (Effective Accelerationism)**
- e/acc, Accelerationism101, Futurology, Futurism, AcceleratingAI
- ArtificialInteligence, OpenAI, LocalLLaMA, Automate, neuralnetworks
- SiliconValley, venturecapital, TheAllinPodcasts, ChatGPT

**EA (Effective Altruism)**
- EffectiveAltruism, slatestarcodex, LessWrong, collapse
- ControlProblem, longevity, ClaudeAI, Anthropic

**Neutral**
- philosophy, programming, technology, singularity, transhumanism

### Time Period

January 1, 2022 - December 31, 2024

### Sentiment Metrics

- **VADER**: Compound score (-1 to +1) optimized for social media
- **TextBlob**: Polarity score (-1 to +1) using lexicon-based approach
- Correlation between methods: ~0.7-0.8

## Key Findings

1. **Alignment Differences**: Significant differences in sentiment across e/acc, EA, and neutral communities (Welch's ANOVA, p < 0.001)

2. **Extreme Sentiment**: e/acc shows higher proportions of both extreme positive and extreme negative sentiment compared to EA

3. **Temporal Patterns**: Sentiment volatility increased following ChatGPT launch (November 2022)

4. **Effect Sizes**: Small to medium Cohen's d values between groups (0.2-0.5)

## Statistical Tests

- Welch's ANOVA (unequal variances)
- Games-Howell post-hoc tests
- Cohen's d effect sizes
- Chi-square tests for extreme sentiment proportions
- Pearson and Spearman correlations
- OLS regression for temporal trends

## Citation

If you use this code or data, please cite:

```
Khan, Y. (2025). Ideology in Silicon Valley: A Sentiment Analysis of AI Discourse 
on Reddit. Oxford Internet Institute, University of Oxford.
```

## License

MIT License

## Contact

**Yasmeena Khan**  
Oxford Internet Institute  
University of Oxford  
Email: yasmeenak@gmail.com  
GitHub: [@yasmeenakhan](https://github.com/yasmeenakhan)