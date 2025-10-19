"""
Reddit scraper for AI discourse analysis across e/acc and EA communities.
"""
import praw
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import os
import random
import config


class RedditScraper:
    """Scrapes posts and comments from Reddit subreddits."""

    def __init__(self):
        """Initialize Reddit API client."""
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )

        self.posts_df = pd.DataFrame()
        self.comments_df = pd.DataFrame()

        self._validate_credentials()

    def _validate_credentials(self):
        """Validate Reddit API credentials."""
        try:
            print("Testing Reddit API connection...")
            try:
                username = self.reddit.user.me()
                print(f"Authenticated as: {username if username else 'read-only'}")
            except:
                print("Authenticated in read-only mode")
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Check your .env file credentials")
            raise SystemExit(1)

    def scrape_subreddit(self, subreddit_name, alignment, limit=None):
        """
        Scrape posts and comments from a subreddit.

        Args:
            subreddit_name: Subreddit to scrape
            alignment: 'eacc', 'ea', or 'neutral'
            limit: Max posts to fetch (None for all)

        Returns:
            Tuple of (posts_df, comments_df)
        """
        print(f"\nScraping r/{subreddit_name}...")

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            _ = subreddit.display_name
        except Exception as e:
            print(f"Error accessing r/{subreddit_name}: {e}")
            return pd.DataFrame(), pd.DataFrame()

        posts_data = []
        comments_data = []
        unique_post_ids = set()

        source_limit = min(1000, limit * 2) if limit else 1000

        sources = {
            'top_all': subreddit.top(limit=source_limit, time_filter='all'),
            'top_year': subreddit.top(limit=source_limit, time_filter='year'),
            'hot': subreddit.hot(limit=source_limit),
            'new': subreddit.new(limit=source_limit)
        }

        for source_name, source_posts in sources.items():
            print(f"Fetching {source_name} posts...")

            with tqdm(total=source_limit, desc=f"{source_name}") as pbar:
                post_count = 0

                for post in source_posts:
                    try:
                        if post.id not in unique_post_ids and \
                           config.START_DATE <= post.created_utc <= config.END_DATE:

                            unique_post_ids.add(post.id)

                            posts_data.append({
                                'post_id': post.id,
                                'post_title': post.title,
                                'post_text': post.selftext,
                                'subreddit': subreddit_name,
                                'alignment': alignment,
                                'author': str(post.author) if post.author else '[deleted]',
                                'created_utc': post.created_utc,
                                'created_date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'url': post.url,
                                'permalink': post.permalink
                            })

                            # Get comments
                            try:
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list():
                                    try:
                                        comments_data.append({
                                            'comment_id': comment.id,
                                            'post_id': post.id,
                                            'parent_id': comment.parent_id,
                                            'comment_text': comment.body,
                                            'subreddit': subreddit_name,
                                            'alignment': alignment,
                                            'author': str(comment.author) if comment.author else '[deleted]',
                                            'created_utc': comment.created_utc,
                                            'created_date': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                                            'score': comment.score
                                        })
                                    except Exception as e:
                                        print(f"Error processing comment: {e}")
                            except Exception as e:
                                print(f"Error processing comments for post {post.id}: {e}")

                            post_count += 1

                    except Exception as e:
                        print(f"Error processing post: {e}")

                    pbar.update(1)
                    time.sleep(0.2 + random.uniform(0, 0.2))

                    if limit and post_count >= limit:
                        break

            if limit and len(posts_data) >= limit:
                break

        posts_data = sorted(posts_data, key=lambda x: x['score'], reverse=True)

        if limit and len(posts_data) > limit:
            posts_data = posts_data[:limit]

        posts_df = pd.DataFrame(posts_data) if posts_data else pd.DataFrame()
        comments_df = pd.DataFrame(comments_data) if comments_data else pd.DataFrame()

        print(f"Scraped {len(posts_df)} posts, {len(comments_df)} comments")

        return posts_df, comments_df

    def scrape_all(self, test_mode=False, test_limit=10):
        """
        Scrape all configured subreddits.

        Args:
            test_mode: If True, only scrape one subreddit from each category
            test_limit: Number of posts per subreddit in test mode
        """
        all_posts = []
        all_comments = []

        if test_mode:
            print(f"\n=== TEST MODE: Scraping {test_limit} posts per category ===")

            # Just one subreddit from each category for testing
            print("\n=== Testing e/acc subreddit ===")
            posts, comments = self.scrape_subreddit(
                config.EACC_SUBREDDITS[0], 'e/acc', limit=test_limit
            )
            if not posts.empty:
                all_posts.append(posts)
            if not comments.empty:
                all_comments.append(comments)

            print("\n=== Testing EA subreddit ===")
            posts, comments = self.scrape_subreddit(
                config.EA_SUBREDDITS[0], 'EA', limit=test_limit
            )
            if not posts.empty:
                all_posts.append(posts)
            if not comments.empty:
                all_comments.append(comments)

            print("\n=== Testing neutral subreddit ===")
            posts, comments = self.scrape_subreddit(
                config.NEUTRAL_SUBREDDITS[0], 'neutral', limit=test_limit
            )
            if not posts.empty:
                all_posts.append(posts)
            if not comments.empty:
                all_comments.append(comments)
        else:
            print("\n=== Scraping e/acc subreddits ===")
            for subreddit in config.EACC_SUBREDDITS:
                posts, comments = self.scrape_subreddit(
                    subreddit, 'e/acc', limit=config.POSTS_LIMIT
                )
                if not posts.empty:
                    all_posts.append(posts)
                if not comments.empty:
                    all_comments.append(comments)

            print("\n=== Scraping EA subreddits ===")
            for subreddit in config.EA_SUBREDDITS:
                posts, comments = self.scrape_subreddit(
                    subreddit, 'EA', limit=config.POSTS_LIMIT
                )
                if not posts.empty:
                    all_posts.append(posts)
                if not comments.empty:
                    all_comments.append(comments)

            print("\n=== Scraping neutral subreddits ===")
            for subreddit in config.NEUTRAL_SUBREDDITS:
                posts, comments = self.scrape_subreddit(
                    subreddit, 'neutral', limit=config.POSTS_LIMIT
                )
                if not posts.empty:
                    all_posts.append(posts)
                if not comments.empty:
                    all_comments.append(comments)

        self.posts_df = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()
        self.comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()

        return self.posts_df, self.comments_df

    def save_data(self):
        """Save scraped data to CSV."""
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

        if not self.posts_df.empty:
            print(f"\nPosts by alignment:")
            print(self.posts_df['alignment'].value_counts())

            self.posts_df.to_csv(config.RAW_POSTS_FILE, index=False)
            print(f"Saved {len(self.posts_df)} posts to {config.RAW_POSTS_FILE}")

        if not self.comments_df.empty:
            print(f"\nComments by alignment:")
            print(self.comments_df['alignment'].value_counts())

            self.comments_df.to_csv(config.RAW_COMMENTS_FILE, index=False)
            print(f"Saved {len(self.comments_df)} comments to {config.RAW_COMMENTS_FILE}")


def main(test_mode=False, test_limit=10):
    """
    Run the scraper.

    Args:
        test_mode: If True, only scrape a few posts for testing
        test_limit: Number of posts per subreddit in test mode
    """
    print("Initializing Reddit scraper...")
    scraper = RedditScraper()

    print("Starting data collection...")
    scraper.scrape_all(test_mode=test_mode, test_limit=test_limit)

    print("\nSaving data...")
    scraper.save_data()

    print("\nScraping complete!")
    print(f"Total posts: {len(scraper.posts_df)}")
    print(f"Total comments: {len(scraper.comments_df)}")


if __name__ == "__main__":
    import sys

    # Test mode: python src/scraper.py test
    # Full mode: python src/scraper.py
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"Running in TEST MODE with limit={test_limit}")
        main(test_mode=True, test_limit=test_limit)
    else:
        print("Running in FULL MODE (this will take hours)")
        main()