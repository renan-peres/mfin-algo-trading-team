# Import additional libraries for sentiment analysis
from textblob import TextBlob
import re
from collections import defaultdict

# ===============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ===============================================================================

def extract_stock_symbols(text, all_tickers=None, excluded_symbols=None):
    """Extract valid stock symbols from text, excluding common words"""
    if all_tickers is None:
        all_tickers = set()
    if excluded_symbols is None:
        excluded_symbols = {'AI', 'S', 'A', 'U', 'E', 'US', 'ET', 'TSXV', 'CODI', 'C'}
    
    symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
    return [symbol for symbol in symbols 
            if symbol in all_tickers and symbol not in excluded_symbols]

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob and classify as bullish/bearish/neutral"""
    polarity = TextBlob(text).sentiment.polarity
    
    if polarity > 0.1:
        return 'bullish', polarity
    elif polarity < -0.1:
        return 'bearish', polarity
    else:
        return 'neutral', polarity

def calculate_stock_sentiment_metrics(df, all_tickers=None, excluded_symbols=None):
    """Calculate comprehensive sentiment metrics for each stock symbol"""
    if all_tickers is None:
        all_tickers = set()
    if excluded_symbols is None:
        excluded_symbols = {'AI', 'S', 'A', 'U', 'E', 'US', 'ET', 'TSXV', 'CODI', 'C'}
    
    stock_metrics = defaultdict(lambda: {
        'sentiment_scores': [],
        'bullish_count': 0,
        'bearish_count': 0,
        'neutral_count': 0,
        'total_articles': 0
    })
    
    # Process each news article
    for row in df.iter_rows(named=True):
        full_text = f"{row.get('title', '')} {row.get('text', '')}"
        mentioned_symbols = extract_stock_symbols(full_text, all_tickers, excluded_symbols)
        sentiment_type, sentiment_score = analyze_sentiment(full_text)
        
        # Update metrics for each mentioned symbol
        for symbol in mentioned_symbols:
            metrics = stock_metrics[symbol]
            metrics['sentiment_scores'].append(sentiment_score)
            metrics['total_articles'] += 1
            metrics[f'{sentiment_type}_count'] += 1
    
    # Calculate final metrics
    final_metrics = {}
    for symbol, data in stock_metrics.items():
        if data['total_articles'] > 0:
            total = data['total_articles']
            avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
            
            final_metrics[symbol] = {
                "articlesInLastWeek": total,
                "companyNewsScore": round((avg_sentiment + 1) / 2, 4),
                "sentiment": {
                    "bearishPercent": round(data['bearish_count'] / total, 4),
                    "bullishPercent": round(data['bullish_count'] / total, 4)
                },
                "averageSentimentScore": round(avg_sentiment, 4),
                "totalArticles": total
            }
    
    return final_metrics

# ===============================================================================
# SECTOR ANALYSIS & FUNDAMENTAL DATA INTEGRATION
# ===============================================================================

def calculate_sector_averages(sentiment_df, fundamentals_pandas):
    """Calculate sector-level sentiment averages"""
    sector_metrics = defaultdict(list)
    
    for row in sentiment_df.iter_rows(named=True):
        symbol = row['symbol']
        if symbol in fundamentals_pandas.index:
            sector = fundamentals_pandas.loc[symbol, 'Sector']
            sector_metrics[sector].append({
                'bullishPercent': row['bullishPercent'],
                'newsScore': row['companyNewsScore']
            })
    
    return {
        sector: {
            'sectorAverageBullishPercent': round(sum(m['bullishPercent'] for m in metrics) / len(metrics), 4),
            'sectorAverageNewsScore': round(sum(m['newsScore'] for m in metrics) / len(metrics), 4)
        }
        for sector, metrics in sector_metrics.items() if metrics
    }

def get_fundamental_value(symbol, column, default=0):
    """Safely get fundamental data value for a symbol"""
    # This function needs access to fundamentals_pandas from the calling scope
    # We'll modify this to accept it as a parameter
    import pandas as pd
    try:
        # Try to access the global fundamentals_pandas if it exists
        import __main__
        if hasattr(__main__, 'fundamentals_pandas'):
            fundamentals_pandas = __main__.fundamentals_pandas
            return fundamentals_pandas.loc[symbol, column] if symbol in fundamentals_pandas.index else default
        else:
            return default
    except:
        return default