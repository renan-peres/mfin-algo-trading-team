import talib
import bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
from itertools import product  
import warnings, logging, os, sys
from io import StringIO
warnings.filterwarnings('ignore')
logging.getLogger('bt').setLevel(logging.CRITICAL + 1)
os.environ['BT_PROGRESS'] = 'False'

# ===============================================================================
# TECHNICAL INDICATORS FRAMEWORK (SIMPLIFIED)
# ===============================================================================

def calculate_technical_indicators(prices_df):
    """
    Calculate selected technical indicators: SMA, EMA, ADX, RSI
    Returns a dictionary of DataFrames with indicators for each stock
    """
    indicators = {}
    
    # Loop through each stock ticker
    for ticker in prices_df.columns:
        print(f"Calculating indicators for {ticker}...")
        
        # Get OHLC data (assuming we only have close prices, we'll use them for all)
        close = prices_df[ticker].dropna()
        high = close  # Simplified - in real scenario you'd have separate OHLC data
        low = close
        
        # Initialize indicator storage for this ticker
        ticker_indicators = pd.DataFrame(index=close.index)
        ticker_indicators['Close'] = close
        
        # ===================================================================
        # SIMPLE MOVING AVERAGES
        # ===================================================================
        ticker_indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        ticker_indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        ticker_indicators['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        # ===================================================================
        # EXPONENTIAL MOVING AVERAGES
        # ===================================================================
        ticker_indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
        ticker_indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
        ticker_indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # ===================================================================
        # AVERAGE DIRECTIONAL INDEX (ADX)
        # ===================================================================
        ticker_indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        ticker_indicators['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        ticker_indicators['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # ===================================================================
        # RELATIVE STRENGTH INDEX (RSI)
        # ===================================================================
        ticker_indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
        ticker_indicators['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # Store the indicators for this ticker
        indicators[ticker] = ticker_indicators
    
    return indicators

def generate_trading_signals(indicators_dict):
    """
    Generate buy/sell signals based on SMA, EMA, ADX, and RSI indicators
    Returns a dictionary of signal DataFrames for each strategy
    """
    strategies = {}
    
    for ticker, indicators in indicators_dict.items():
        ticker_signals = pd.DataFrame(index=indicators.index)
        
        # ===================================================================
        # STRATEGY 1: SMA CROSSOVER
        # ===================================================================
        ticker_signals['SMA_Cross_Signal'] = np.where(
            indicators['SMA_20'] > indicators['SMA_50'], 1, 
            np.where(indicators['SMA_20'] < indicators['SMA_50'], -1, 0)
        )
        
        # ===================================================================
        # STRATEGY 2: EMA CROSSOVER
        # ===================================================================
        ticker_signals['EMA_Cross_Signal'] = np.where(
            indicators['EMA_12'] > indicators['EMA_26'], 1, 
            np.where(indicators['EMA_12'] < indicators['EMA_26'], -1, 0)
        )
        
        # ===================================================================
        # STRATEGY 3: ADX TREND STRENGTH
        # ===================================================================
        # Strong trend: ADX > 25, buy when +DI > -DI, sell when -DI > +DI
        ticker_signals['ADX_Trend_Signal'] = np.where(
            (indicators['ADX'] > 25) & (indicators['PLUS_DI'] > indicators['MINUS_DI']), 1,
            np.where((indicators['ADX'] > 25) & (indicators['MINUS_DI'] > indicators['PLUS_DI']), -1, 0)
        )
        
        # ===================================================================
        # STRATEGY 4: RSI MEAN REVERSION
        # ===================================================================
        ticker_signals['RSI_Signal'] = np.where(
            indicators['RSI_14'] < 30, 1,  # Oversold - Buy
            np.where(indicators['RSI_14'] > 70, -1, 0)  # Overbought - Sell
        )
        
        strategies[ticker] = ticker_signals
    
    return strategies

import talib
import bt
import pandas as pd
import numpy as np
import warnings, logging, os, sys
from io import StringIO

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger('bt').setLevel(logging.CRITICAL + 1)
os.environ['BT_PROGRESS'] = 'False'

# ===============================================================================
# CONSTANTS
# ===============================================================================
STRATEGY_NAMES = ['SMA_Cross_Signal', 'EMA_Cross_Signal', 'ADX_Trend_Signal', 'RSI_Signal']

# ===============================================================================
# TECHNICAL INDICATORS FRAMEWORK
# ===============================================================================

def calculate_technical_indicators(prices_df):
    """
    Calculate selected technical indicators: SMA, EMA, ADX, RSI
    Returns a dictionary of DataFrames with indicators for each stock
    """
    indicators = {}
    
    for ticker in prices_df.columns:       
        close = prices_df[ticker].dropna()
        if len(close) < 50:  # Skip if insufficient data
            print(f"  ⚠️ Skipping {ticker}: insufficient data ({len(close)} points)")
            continue
            
        high = close  # Simplified - using close as proxy for OHLC
        low = close
        
        ticker_indicators = pd.DataFrame(index=close.index)
        ticker_indicators['Close'] = close
        
        try:
            # Simple Moving Averages
            ticker_indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
            ticker_indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
            ticker_indicators['SMA_200'] = talib.SMA(close, timeperiod=200)
            
            # Exponential Moving Averages
            ticker_indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
            ticker_indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
            ticker_indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
            
            # Average Directional Index (ADX)
            ticker_indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            ticker_indicators['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            ticker_indicators['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Relative Strength Index (RSI)
            ticker_indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
            ticker_indicators['RSI_21'] = talib.RSI(close, timeperiod=21)
            
            indicators[ticker] = ticker_indicators
            
        except Exception as e:
            print(f"  ❌ Error calculating indicators for {ticker}: {str(e)}")
            continue
    
    print(f"✅ Technical indicators calculated for {len(indicators)} tickers")
    return indicators

def generate_trading_signals(indicators_dict):
    """
    Generate buy/sell signals based on SMA, EMA, ADX, and RSI indicators
    Returns a dictionary of signal DataFrames for each strategy
    """
    strategies = {}
    
    for ticker, indicators in indicators_dict.items():
        try:
            ticker_signals = pd.DataFrame(index=indicators.index)
            
            # Strategy 1: SMA Crossover
            ticker_signals['SMA_Cross_Signal'] = np.where(
                indicators['SMA_20'] > indicators['SMA_50'], 1, 
                np.where(indicators['SMA_20'] < indicators['SMA_50'], -1, 0)
            )
            
            # Strategy 2: EMA Crossover
            ticker_signals['EMA_Cross_Signal'] = np.where(
                indicators['EMA_12'] > indicators['EMA_26'], 1, 
                np.where(indicators['EMA_12'] < indicators['EMA_26'], -1, 0)
            )
            
            # Strategy 3: ADX Trend Strength
            ticker_signals['ADX_Trend_Signal'] = np.where(
                (indicators['ADX'] > 25) & (indicators['PLUS_DI'] > indicators['MINUS_DI']), 1,
                np.where((indicators['ADX'] > 25) & (indicators['MINUS_DI'] > indicators['PLUS_DI']), -1, 0)
            )
            
            # Strategy 4: RSI Mean Reversion
            ticker_signals['RSI_Signal'] = np.where(
                indicators['RSI_14'] < 30, 1,  # Oversold - Buy
                np.where(indicators['RSI_14'] > 70, -1, 0)  # Overbought - Sell
            )
            
            strategies[ticker] = ticker_signals
            
        except Exception as e:
            print(f"❌ Error generating signals for {ticker}: {str(e)}")
            continue
    
    print(f"✅ Trading signals generated for {len(strategies)} tickers")
    return strategies

# ===============================================================================
# BACKTESTING FRAMEWORK
# ===============================================================================

def create_bt_strategy(signal_name, target_weights):
    """Create a bt strategy with target weights and rebalancing."""
    return bt.Strategy(signal_name, [bt.algos.WeighTarget(target_weights), bt.algos.Rebalance()])

def safe_get_stat(stats, strategy_col, stat_names, default=0):
    """Safely extract statistics with fallback options."""
    for stat_name in stat_names:
        if stat_name in stats.index:
            value = stats.loc[stat_name, strategy_col]
            return value if not pd.isna(value) else default
    return default

def run_backtest_silent(backtest):
    """Run backtest while suppressing output."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = StringIO()
        return bt.run(backtest)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def extract_performance_metrics(result, strategy_name):
    """Extract key performance metrics from backtest result."""
    try:
        stats = result.stats
        
        return {
            'result': result,
            'total_return': safe_get_stat(stats, strategy_name, ['total_return']),
            'sharpe_ratio': safe_get_stat(stats, strategy_name, ['daily_sharpe', 'monthly_sharpe', 'yearly_sharpe']),
            'max_drawdown': safe_get_stat(stats, strategy_name, ['max_drawdown']),
            'volatility': safe_get_stat(stats, strategy_name, ['daily_vol', 'monthly_vol', 'yearly_vol']),
            'cagr': safe_get_stat(stats, strategy_name, ['cagr'])
        }
    except Exception as e:
        print(f"⚠️ Error extracting metrics for {strategy_name}: {str(e)}")
        return None

def get_portfolio_stats(portfolio_result):
    """Extract portfolio statistics from backtest result."""
    try:
        stats = portfolio_result.stats
        strategy_col = stats.columns[0]
        return {
            'cagr': safe_get_stat(stats, strategy_col, ['cagr']),
            'total_return': safe_get_stat(stats, strategy_col, ['total_return']),
            'sharpe_ratio': safe_get_stat(stats, strategy_col, ['daily_sharpe', 'monthly_sharpe', 'yearly_sharpe']),
            'max_drawdown': safe_get_stat(stats, strategy_col, ['max_drawdown']),
            'volatility': safe_get_stat(stats, strategy_col, ['daily_vol', 'monthly_vol', 'yearly_vol'])
        }
    except Exception as e:
        print(f"⚠️ Error extracting portfolio stats: {str(e)}")
        return {'cagr': 0, 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

def backtest_strategies_silent(quotes, trading_signals, available_tickers, strategy_names=None):
    """Run backtests for all strategies and tickers silently."""
    if strategy_names is None: 
        strategy_names = STRATEGY_NAMES
    
    results = {}
    total_tests = len(available_tickers) * len(strategy_names)
    print(f"🔄 Running {total_tests} backtests silently...")
    
    completed = 0
    for i, ticker in enumerate(available_tickers):
        results[ticker] = {}
        
        try:
            ticker_data = quotes[[ticker]].dropna()
            if len(ticker_data) < 50:  # Skip if insufficient data
                print(f"  ⚠️ Skipping {ticker}: insufficient data")
                for strategy_name in strategy_names:
                    results[ticker][strategy_name] = None
                continue
        except Exception as e:
            print(f"  ❌ Error accessing data for {ticker}: {str(e)}")
            for strategy_name in strategy_names:
                results[ticker][strategy_name] = None
            continue
        
        for j, strategy_name in enumerate(strategy_names):
            try:
                if ticker not in trading_signals or strategy_name not in trading_signals[ticker]:
                    results[ticker][strategy_name] = None
                    continue
                    
                signals = trading_signals[ticker][strategy_name].reindex(ticker_data.index, method='ffill')
                target_weights = pd.DataFrame(index=ticker_data.index, columns=[ticker])
                target_weights[ticker] = signals
                
                strategy = create_bt_strategy(f"{ticker}_{strategy_name}", target_weights)
                backtest = bt.Backtest(strategy, ticker_data)
                result = run_backtest_silent(backtest)
                results[ticker][strategy_name] = extract_performance_metrics(result, strategy.name)
                
                completed += 1
                if completed % 20 == 0:  # Progress every 20 tests
                    print(f"  Progress: {(completed / total_tests) * 100:.0f}%")
                    
            except Exception as e:
                print(f"  ❌ Error backtesting {ticker}-{strategy_name}: {str(e)}")
                results[ticker][strategy_name] = None
    
    print("✅ Backtesting completed")
    return results

def find_best_strategies(backtest_results, metric='cagr'):
    """Find the best performing strategy for each ticker based on specified metric."""
    best_strategies = {}
    
    for ticker, ticker_results in backtest_results.items():
        best_strategy, best_value = None, -np.inf
        
        for strategy_name, result in ticker_results.items():
            if (result and isinstance(result, dict) and metric in result 
                and not np.isnan(result[metric]) and result[metric] > best_value):
                best_value, best_strategy = result[metric], strategy_name
        
        if best_strategy:
            best_result = backtest_results[ticker][best_strategy]
            best_strategies[ticker] = {
                k: best_result[k] for k in ['cagr', 'total_return', 'max_drawdown', 'sharpe_ratio', 'volatility']
            }
            best_strategies[ticker]['strategy'] = best_strategy
        else:
            best_strategies[ticker] = {
                'strategy': None, 'cagr': 0, 'total_return': 0, 
                'max_drawdown': 0, 'sharpe_ratio': 0, 'volatility': 0
            }
    
    return best_strategies

def create_strategy_comparison(backtest_results, available_tickers, strategy_names=None):
    """Create comparison table of strategy performance across all tickers."""
    if strategy_names is None: 
        strategy_names = STRATEGY_NAMES
    
    comparison_data = []
    
    for strategy_name in strategy_names:
        valid_results = [
            backtest_results[ticker][strategy_name] 
            for ticker in available_tickers 
            if (ticker in backtest_results 
                and strategy_name in backtest_results[ticker] 
                and backtest_results[ticker][strategy_name] is not None
                and isinstance(backtest_results[ticker][strategy_name], dict))
        ]
        
        if valid_results:
            safe_mean = lambda values: np.mean([v for v in values if not np.isnan(v)]) if values else 0
            comparison_data.append({
                'Strategy': strategy_name,
                'Avg_CAGR': safe_mean([r['cagr'] for r in valid_results]),
                'Avg_Return': safe_mean([r['total_return'] for r in valid_results]),
                'Avg_Sharpe': safe_mean([r['sharpe_ratio'] for r in valid_results]),
                'Avg_Volatility': safe_mean([r['volatility'] for r in valid_results]),
                'Success_Rate': len(valid_results) / len(available_tickers),
                'Valid_Tests': len(valid_results)
            })
        else:
            comparison_data.append({
                'Strategy': strategy_name, 
                'Avg_CAGR': 0, 
                'Avg_Return': 0, 
                'Avg_Sharpe': 0, 
                'Avg_Volatility': 0, 
                'Success_Rate': 0, 
                'Valid_Tests': 0
            })
    
    return pd.DataFrame(comparison_data).sort_values('Avg_CAGR', ascending=False)

def create_equal_weight_portfolio(quotes, trading_signals, best_strategies, available_tickers, 
                                min_cagr=0.1, max_volatility=0.3, max_stocks=10):
    """Create an equal-weight portfolio from filtered top-performing strategies."""
    
    # Filter by CAGR requirement
    cagr_filtered = [
        ticker for ticker in available_tickers 
        if (ticker in best_strategies 
            and best_strategies[ticker]['strategy'] 
            and best_strategies[ticker]['cagr'] >= min_cagr)
    ]
    
    # Filter by volatility requirement
    filtered_tickers = [
        ticker for ticker in cagr_filtered
        if best_strategies[ticker]['volatility'] <= max_volatility
    ]
    
    print(f"📊 CAGR Filtering: {len(available_tickers)} → {len(cagr_filtered)} tickers (≥{min_cagr:.1%})")
    print(f"📊 Volatility Filtering: → {len(filtered_tickers)} tickers (≤{max_volatility:.1%})")
    
    if not filtered_tickers:
        print("⚠️ No tickers meet minimum CAGR and maximum volatility requirements!")
        return pd.DataFrame(index=quotes.index, columns=available_tickers).fillna(0), []
    
    # Select top performers by CAGR
    top_tickers = sorted(filtered_tickers, key=lambda x: best_strategies[x]['cagr'], reverse=True)[:max_stocks]
    
    print(f"🏆 Selected top {len(top_tickers)} performers:")
    for i, ticker in enumerate(top_tickers, 1):
        strategy_info = best_strategies[ticker]
        print(f"  #{i}. {ticker}: {strategy_info['cagr']:.2%} CAGR, "
              f"{strategy_info['volatility']:.2%} Vol, "
              f"Sharpe: {strategy_info['sharpe_ratio']:.2f} "
              f"({strategy_info['strategy']})")
    
    # Create portfolio weights
    weight_per_ticker = 1.0 / len(top_tickers)
    portfolio_weights = pd.DataFrame(index=quotes.index, columns=available_tickers).fillna(0)
    
    for ticker in top_tickers:
        try:
            if ticker in trading_signals:
                strategy_name = best_strategies[ticker]['strategy']
                if strategy_name in trading_signals[ticker]:
                    signals = trading_signals[ticker][strategy_name].reindex(quotes.index, method='ffill')
                    portfolio_weights[ticker] = signals * weight_per_ticker
                else:
                    print(f"⚠️ Strategy {strategy_name} not found for {ticker}")
            else:
                print(f"⚠️ No trading signals found for {ticker}")
        except Exception as e:
            print(f"❌ Error creating weights for {ticker}: {str(e)}")
    
    return portfolio_weights, top_tickers

# ===============================================================================
# PORTFOLIO ANALYSIS FRAMEWORK
# ===============================================================================

def run_backtest(quotes, trading_signals, available_tickers):
    """
    Run strategy backtesting and return basic results.
    """
    print("=" * 80)
    print("STRATEGY BACKTESTING ANALYSIS")
    print("=" * 80)
    
    # Phase 1: Strategy Backtesting
    print("\n🔄 Running strategy backtests...")
    backtest_results = backtest_strategies_silent(quotes, trading_signals, available_tickers)
    best_strategies = find_best_strategies(backtest_results)
    strategy_comparison = create_strategy_comparison(backtest_results, available_tickers, STRATEGY_NAMES)
    
    return {
        'backtest_results': backtest_results,
        'best_strategies': best_strategies,
        'strategy_comparison': strategy_comparison
    }

def best_strategy_analysis(quotes, trading_signals, backtest_data, available_tickers,
                          min_cagr=0.0, max_volatility=0.3, max_stocks=10):
    """
    Analyze best strategies, construct portfolio, and return complete results.
    """
    best_strategies = backtest_data['best_strategies']
    
    # Phase 2: Portfolio Construction
    print("\n🔄 Portfolio Construction")
    portfolio_weights, selected_tickers = create_equal_weight_portfolio(
        quotes, trading_signals, best_strategies, available_tickers, 
        min_cagr=min_cagr, max_volatility=max_volatility, max_stocks=max_stocks
    )
    
    # Phase 3: Portfolio Backtesting
    print("\n🔄 Portfolio Backtesting")
    portfolio_strategy = create_bt_strategy('OptimalPortfolio', portfolio_weights)
    portfolio_result = run_backtest_silent(bt.Backtest(portfolio_strategy, quotes))
    portfolio_stats = get_portfolio_stats(portfolio_result)
    
    # Display portfolio analysis
    if not selected_tickers:
        print("\n⚠️ No tickers met the minimum CAGR and maximum volatility requirements")
    else:
        # Create analysis DataFrame
        included_df = pd.DataFrame.from_dict(
            {ticker: best_strategies[ticker] for ticker in selected_tickers}, 
            orient='index'
        )
        
        # Calculate portfolio metrics
        avg_cagr = included_df['cagr'].mean()
        avg_sharpe = included_df['sharpe_ratio'].mean()
        avg_volatility = included_df['volatility'].mean()
        avg_drawdown = included_df['max_drawdown'].mean()
        strategy_distribution = included_df['strategy'].value_counts()
    
    # Display top performing strategies for selected tickers only
    if selected_tickers:
        selected_best_strategies = {ticker: best_strategies[ticker] for ticker in selected_tickers if ticker in best_strategies}
        best_strategies_df = pd.DataFrame.from_dict(selected_best_strategies, orient='index')
    else:
        print("\n⚠️ No tickers were selected for the portfolio")
    
    # Display results
    print("✅ Strategy Analysis Complete!")
    print(f"\n📈 Portfolio: {len(selected_tickers)} stocks")
    print(f"  📊 Avg CAGR: {avg_cagr:.2%} | "
            f"Avg Sharpe: {avg_sharpe:.2f} | "
            f"Avg Volatility: {avg_volatility:.2%} | "
            f"Avg Drawdown: {avg_drawdown:.2%}")
    
    print(f"  🎯 Strategy Distribution:")
    for strategy, count in strategy_distribution.items():
        percentage = count / len(included_df) * 100
        print(f"    - {strategy}: {count} stocks ({percentage:.1f}%)")
    
    return {
        'portfolio_result': portfolio_result,
        'best_strategies': best_strategies,
        'best_strategies_df': best_strategies_df if 'best_strategies_df' in locals() else None,
        'strategy_comparison': backtest_data['strategy_comparison'],
        'portfolio_stats': portfolio_stats,
        'portfolio_weights': portfolio_weights,
        'included_tickers': selected_tickers,
        'selected_tickers': selected_tickers,  # Backward compatibility
        'backtest_results': backtest_data['backtest_results'],
        'cagr_threshold': min_cagr,
        'max_volatility': max_volatility,
        'max_stocks': max_stocks
    }

# ===============================================================================
# OPTIMIZE STRATEGY PARAMETERS
# ===============================================================================


def optimize_strategy_parameters(quotes, best_strategies_df, selected_tickers, risk_free_rate=0.0433):
    """
    Optimize parameters for each strategy type based on the best performing strategies
    """
    optimization_results = {}
    
    # Define parameter ranges for each strategy
    parameter_ranges = {
        'SMA_Cross_Signal': {
            'short_periods': range(5, 25, 2),   # 5, 7, 9, ..., 23
            'long_periods': range(20, 55, 5)    # 20, 25, 30, ..., 50
        },
        'EMA_Cross_Signal': {
            'short_periods': range(8, 20, 2),   # 8, 10, 12, ..., 18
            'long_periods': range(21, 35, 3)    # 21, 24, 27, 30, 33
        },
        'RSI_Signal': {
            'rsi_periods': range(10, 25, 2),    # 10, 12, 14, ..., 22
            'oversold_levels': range(25, 35, 2), # 25, 27, 29, 31, 33
            'overbought_levels': range(65, 75, 2) # 65, 67, 69, 71, 73
        },
        'ADX_Trend_Signal': {
            'adx_periods': range(10, 20, 2),    # 10, 12, 14, 16, 18
            'adx_thresholds': range(20, 35, 5)  # 20, 25, 30
        }
    }
    
    for ticker in selected_tickers:
        if ticker not in best_strategies_df.index:
            print(f"⚠️ Skipping {ticker}: not found in best_strategies_df")
            continue
            
        best_strategy = best_strategies_df.loc[ticker, 'strategy']
        print(f"\n🔄 Optimizing {best_strategy} parameters for {ticker}...")
        
        try:
            ticker_data = quotes[[ticker]].dropna()
            if len(ticker_data) < 100:
                print(f"  ⚠️ Insufficient data for {ticker}")
                continue
                
            if best_strategy == 'SMA_Cross_Signal':
                results = optimize_sma_parameters(ticker, ticker_data, parameter_ranges['SMA_Cross_Signal'], risk_free_rate)
            elif best_strategy == 'EMA_Cross_Signal':
                results = optimize_ema_parameters(ticker, ticker_data, parameter_ranges['EMA_Cross_Signal'], risk_free_rate)
            elif best_strategy == 'RSI_Signal':
                results = optimize_rsi_parameters(ticker, ticker_data, parameter_ranges['RSI_Signal'], risk_free_rate)
            elif best_strategy == 'ADX_Trend_Signal':
                results = optimize_adx_parameters(ticker, ticker_data, parameter_ranges['ADX_Trend_Signal'], risk_free_rate)
            else:
                print(f"  ⚠️ Unknown strategy: {best_strategy}")
                continue
                
            optimization_results[ticker] = results
            
        except Exception as e:
            print(f"  ❌ Error optimizing {ticker}: {str(e)}")
            continue
    
    return optimization_results

def optimize_sma_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize SMA crossover parameters"""
    results = []
    
    for short_period, long_period in product(param_ranges['short_periods'], param_ranges['long_periods']):
        if short_period >= long_period:  # Skip invalid combinations
            continue
            
        try:
            # Calculate SMAs
            sma_short = data[ticker].rolling(short_period).mean()
            sma_long = data[ticker].rolling(long_period).mean()
            
            # Generate signals
            target_weights = pd.DataFrame(index=data.index, columns=[ticker])
            target_weights[ticker] = np.where(sma_short > sma_long, 1.0, 
                                            np.where(sma_short < sma_long, -1.0, 0.0))
            target_weights = target_weights.fillna(0.0)
            
            # Create and run backtest
            strategy_name = f'SMA_{short_period}_{long_period}'
            strategy = bt.Strategy(strategy_name, [
                bt.algos.WeighTarget(target_weights),
                bt.algos.Rebalance()
            ])
            
            backtest = bt.Backtest(strategy, data)
            result = bt.run(backtest)
            
            # Extract metrics
            stats = result.stats
            cagr = stats.loc['cagr', strategy_name] if 'cagr' in stats.index else 0
            sharpe = stats.loc['daily_sharpe', strategy_name] if 'daily_sharpe' in stats.index else 0
            max_dd = stats.loc['max_drawdown', strategy_name] if 'max_drawdown' in stats.index else 0
            volatility = stats.loc['daily_vol', strategy_name] if 'daily_vol' in stats.index else 0
            
            results.append({
                'short_period': short_period,
                'long_period': long_period,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'strategy_name': strategy_name
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"  ✅ Best SMA params for {ticker}: {best_params['short_period']}/{best_params['long_period']} "
              f"(Sharpe: {best_params['sharpe_ratio']:.3f})")
        return {'results_df': results_df, 'best_params': best_params, 'strategy_type': 'SMA_Cross_Signal'}
    return None

def optimize_ema_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize EMA crossover parameters"""
    results = []
    
    for short_period, long_period in product(param_ranges['short_periods'], param_ranges['long_periods']):
        if short_period >= long_period:
            continue
            
        try:
            # Calculate EMAs
            ema_short = data[ticker].ewm(span=short_period).mean()
            ema_long = data[ticker].ewm(span=long_period).mean()
            
            # Generate signals
            target_weights = pd.DataFrame(index=data.index, columns=[ticker])
            target_weights[ticker] = np.where(ema_short > ema_long, 1.0,
                                            np.where(ema_short < ema_long, -1.0, 0.0))
            target_weights = target_weights.fillna(0.0)
            
            # Create and run backtest
            strategy_name = f'EMA_{short_period}_{long_period}'
            strategy = bt.Strategy(strategy_name, [
                bt.algos.WeighTarget(target_weights),
                bt.algos.Rebalance()
            ])
            
            backtest = bt.Backtest(strategy, data)
            result = bt.run(backtest)
            
            # Extract metrics
            stats = result.stats
            cagr = stats.loc['cagr', strategy_name] if 'cagr' in stats.index else 0
            sharpe = stats.loc['daily_sharpe', strategy_name] if 'daily_sharpe' in stats.index else 0
            max_dd = stats.loc['max_drawdown', strategy_name] if 'max_drawdown' in stats.index else 0
            volatility = stats.loc['daily_vol', strategy_name] if 'daily_vol' in stats.index else 0
            
            results.append({
                'short_period': short_period,
                'long_period': long_period,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'strategy_name': strategy_name
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"  ✅ Best EMA params for {ticker}: {best_params['short_period']}/{best_params['long_period']} "
              f"(Sharpe: {best_params['sharpe_ratio']:.3f})")
        return {'results_df': results_df, 'best_params': best_params, 'strategy_type': 'EMA_Cross_Signal'}
    return None

def optimize_rsi_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize RSI parameters"""
    import talib
    results = []
    
    for rsi_period, oversold, overbought in product(
        param_ranges['rsi_periods'], 
        param_ranges['oversold_levels'], 
        param_ranges['overbought_levels']
    ):
        if oversold >= overbought:  # Skip invalid combinations
            continue
            
        try:
            # Calculate RSI
            rsi = talib.RSI(data[ticker].values, timeperiod=rsi_period)
            rsi_series = pd.Series(rsi, index=data.index)
            
            # Generate signals
            target_weights = pd.DataFrame(index=data.index, columns=[ticker])
            target_weights[ticker] = np.where(rsi_series < oversold, 1.0,  # Oversold - Buy
                                            np.where(rsi_series > overbought, -1.0, 0.0))  # Overbought - Sell
            target_weights = target_weights.fillna(0.0)
            
            # Create and run backtest
            strategy_name = f'RSI_{rsi_period}_{oversold}_{overbought}'
            strategy = bt.Strategy(strategy_name, [
                bt.algos.WeighTarget(target_weights),
                bt.algos.Rebalance()
            ])
            
            backtest = bt.Backtest(strategy, data)
            result = bt.run(backtest)
            
            # Extract metrics
            stats = result.stats
            cagr = stats.loc['cagr', strategy_name] if 'cagr' in stats.index else 0
            sharpe = stats.loc['daily_sharpe', strategy_name] if 'daily_sharpe' in stats.index else 0
            max_dd = stats.loc['max_drawdown', strategy_name] if 'max_drawdown' in stats.index else 0
            volatility = stats.loc['daily_vol', strategy_name] if 'daily_vol' in stats.index else 0
            
            results.append({
                'rsi_period': rsi_period,
                'oversold_level': oversold,
                'overbought_level': overbought,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'strategy_name': strategy_name
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"  ✅ Best RSI params for {ticker}: Period {best_params['rsi_period']}, "
              f"Levels {best_params['oversold_level']}/{best_params['overbought_level']} "
              f"(Sharpe: {best_params['sharpe_ratio']:.3f})")
        return {'results_df': results_df, 'best_params': best_params, 'strategy_type': 'RSI_Signal'}
    return None

def optimize_adx_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize ADX parameters"""
    import talib
    results = []
    
    for adx_period, adx_threshold in product(param_ranges['adx_periods'], param_ranges['adx_thresholds']):
        try:
            # Calculate ADX and DI indicators
            high = low = close = data[ticker].values  # Using close as proxy
            adx = talib.ADX(high, low, close, timeperiod=adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=adx_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=adx_period)
            
            adx_series = pd.Series(adx, index=data.index)
            plus_di_series = pd.Series(plus_di, index=data.index)
            minus_di_series = pd.Series(minus_di, index=data.index)
            
            # Generate signals
            target_weights = pd.DataFrame(index=data.index, columns=[ticker])
            target_weights[ticker] = np.where(
                (adx_series > adx_threshold) & (plus_di_series > minus_di_series), 1.0,
                np.where((adx_series > adx_threshold) & (minus_di_series > plus_di_series), -1.0, 0.0)
            )
            target_weights = target_weights.fillna(0.0)
            
            # Create and run backtest
            strategy_name = f'ADX_{adx_period}_{adx_threshold}'
            strategy = bt.Strategy(strategy_name, [
                bt.algos.WeighTarget(target_weights),
                bt.algos.Rebalance()
            ])
            
            backtest = bt.Backtest(strategy, data)
            result = bt.run(backtest)
            
            # Extract metrics
            stats = result.stats
            cagr = stats.loc['cagr', strategy_name] if 'cagr' in stats.index else 0
            sharpe = stats.loc['daily_sharpe', strategy_name] if 'daily_sharpe' in stats.index else 0
            max_dd = stats.loc['max_drawdown', strategy_name] if 'max_drawdown' in stats.index else 0
            volatility = stats.loc['daily_vol', strategy_name] if 'daily_vol' in stats.index else 0
            
            results.append({
                'adx_period': adx_period,
                'adx_threshold': adx_threshold,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': volatility,
                'strategy_name': strategy_name
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"  ✅ Best ADX params for {ticker}: Period {best_params['adx_period']}, "
              f"Threshold {best_params['adx_threshold']} "
              f"(Sharpe: {best_params['sharpe_ratio']:.3f})")
        return {'results_df': results_df, 'best_params': best_params, 'strategy_type': 'ADX_Trend_Signal'}
    return None

def plot_optimization_heatmaps(optimization_results):
    """Plot heatmaps for parameter optimization results"""
    for ticker, results in optimization_results.items():
        if results is None:
            continue
            
        results_df = results['results_df']
        strategy_type = results['strategy_type']
        
        plt.figure(figsize=(12, 8))
        
        if strategy_type in ['SMA_Cross_Signal', 'EMA_Cross_Signal']:
            # Create pivot table for crossover strategies
            pivot_table = results_df.pivot_table(
                values='sharpe_ratio', 
                index='long_period', 
                columns='short_period', 
                fill_value=0
            )
            
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, cbar_kws={'label': 'Sharpe Ratio'})
            plt.title(f'{ticker} - {strategy_type} Parameter Optimization\nSharpe Ratio Heatmap')
            plt.xlabel('Short Period')
            plt.ylabel('Long Period')
            
        elif strategy_type == 'RSI_Signal':
            # Create separate heatmaps for RSI
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Heatmap 1: RSI Period vs Oversold Level
            pivot1 = results_df.pivot_table(
                values='sharpe_ratio',
                index='oversold_level',
                columns='rsi_period',
                fill_value=0
            )
            sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, ax=ax1, cbar_kws={'label': 'Sharpe Ratio'})
            ax1.set_title(f'{ticker} - RSI Period vs Oversold Level')
            ax1.set_xlabel('RSI Period')
            ax1.set_ylabel('Oversold Level')
            
            # Heatmap 2: RSI Period vs Overbought Level
            pivot2 = results_df.pivot_table(
                values='sharpe_ratio',
                index='overbought_level',
                columns='rsi_period',
                fill_value=0
            )
            sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, ax=ax2, cbar_kws={'label': 'Sharpe Ratio'})
            ax2.set_title(f'{ticker} - RSI Period vs Overbought Level')
            ax2.set_xlabel('RSI Period')
            ax2.set_ylabel('Overbought Level')
            
        elif strategy_type == 'ADX_Trend_Signal':
            # Heatmap for ADX
            pivot_table = results_df.pivot_table(
                values='sharpe_ratio',
                index='adx_threshold',
                columns='adx_period',
                fill_value=0
            )
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, cbar_kws={'label': 'Sharpe Ratio'})
            plt.title(f'{ticker} - ADX Parameter Optimization\nSharpe Ratio Heatmap')
            plt.xlabel('ADX Period')
            plt.ylabel('ADX Threshold')
        
        plt.tight_layout()
        plt.show()

def create_optimization_summary(optimization_results):
    """Create a summary DataFrame of optimization results"""
    summary_data = []
    
    for ticker, results in optimization_results.items():
        if results is None:
            continue
            
        best_params = results['best_params']
        strategy_type = results['strategy_type']
        
        summary_row = {
            'Ticker': ticker,
            'Strategy': strategy_type,
            'Best_Sharpe': best_params['sharpe_ratio'],
            'CAGR': best_params['cagr'],
            'Max_Drawdown': best_params['max_drawdown'],
            'Volatility': best_params['volatility']
        }
        
        # Add strategy-specific parameters
        if strategy_type in ['SMA_Cross_Signal', 'EMA_Cross_Signal']:
            summary_row['Short_Period'] = best_params['short_period']
            summary_row['Long_Period'] = best_params['long_period']
        elif strategy_type == 'RSI_Signal':
            summary_row['RSI_Period'] = best_params['rsi_period']
            summary_row['Oversold_Level'] = best_params['oversold_level']
            summary_row['Overbought_Level'] = best_params['overbought_level']
        elif strategy_type == 'ADX_Trend_Signal':
            summary_row['ADX_Period'] = best_params['adx_period']
            summary_row['ADX_Threshold'] = best_params['adx_threshold']
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

# ===============================================================================
# PLOTTING FRAMEWORK
# ===============================================================================

def plot_ticker_signals_with_annotations(quotes, trading_signals, technical_indicators, best_strategies, included_tickers):
    """Plot individual ticker charts with signals and technical indicators for included tickers only"""
    if not included_tickers:
        print("❌ No tickers available for plotting (none passed the CAGR and volatility thresholds)")
        return
    
    print(f"📊 Plotting {len(included_tickers)} tickers that passed the 10% CAGR and ≤30% volatility thresholds")
    
    cols = min(2, len(included_tickers))
    rows = (len(included_tickers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
    axes_flat = [axes] if len(included_tickers) == 1 else axes.flatten()
    
    def plot_signals(ax, signals, price_data, strategy_name, indicators=None, ax2=None):
        """Helper function to plot buy/sell signals with annotations"""
        if strategy_name not in signals.columns:
            return ""
        
        strategy_signals = signals[strategy_name]
        signal_changes = strategy_signals.diff()
        
        # Signal configuration
        signal_configs = [(1, lambda x: x > 0, 'green', '^', 'Long'), (-1, lambda x: x < 0, 'red', 'v', 'Short')]
        
        # Plot signals based on strategy type
        if strategy_name == 'RSI_Signal' and ax2 and indicators is not None and 'RSI_14' in indicators.columns:
            target_ax, y_source, offset = ax2, 'RSI_14', (10, 5)
        elif strategy_name in ['SMA_Cross_Signal', 'EMA_Cross_Signal'] and indicators is not None:
            ma_configs = {'SMA_Cross_Signal': 'SMA_20', 'EMA_Cross_Signal': 'EMA_12'}
            fast_ma = ma_configs[strategy_name]
            if fast_ma in indicators.columns:
                target_ax, y_source, offset = ax, fast_ma, (10, 20)
            else:
                target_ax, y_source, offset = ax, 'price', (10, 20)
        else:
            target_ax, y_source, offset = ax, 'price', (10, 20)
        
        for signal_type, change_val, color, marker, text in signal_configs:
            crossovers = signal_changes[(change_val(signal_changes)) & (strategy_signals == signal_type)]
            for date in crossovers.index:
                if y_source == 'price' and date in price_data.index:
                    y_val = price_data[date]
                elif y_source != 'price' and date in indicators.index:
                    y_val = indicators.loc[date, y_source]
                else:
                    continue
                
                target_ax.scatter(date, y_val, color=color, marker=marker, s=60 if y_source == 'RSI_14' else 80, alpha=0.9, zorder=5)
                target_ax.annotate(text, xy=(date, y_val), xytext=offset, textcoords='offset points',
                                 fontsize=8 if y_source == 'RSI_14' else 9, fontweight='bold', color=color, ha='left',
                                 va='bottom' if signal_type == 1 else 'top',
                                 bbox=dict(boxstyle='round,pad=0.2' if y_source == 'RSI_14' else 'round,pad=0.3', 
                                          facecolor='white', edgecolor=color, alpha=0.8 if y_source == 'RSI_14' else 0.9),
                                 arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
        
        buy_count = len(signal_changes[(signal_changes > 0) & (strategy_signals == 1)])
        sell_count = len(signal_changes[(signal_changes < 0) & (strategy_signals == -1)])
        return f"Signals: {buy_count} Long, {sell_count} Short"
    
    def plot_indicators(ax, indicators, strategy):
        """Helper function to plot strategy-specific indicators"""
        ax2 = None
        indicator_configs = {
            'SMA_Cross_Signal': [('SMA_20', 'SMA 20', 'blue'), ('SMA_50', 'SMA 50', 'orange')],
            'EMA_Cross_Signal': [('EMA_12', 'EMA 12', 'green'), ('EMA_26', 'EMA 26', 'red')]
        }
        
        if strategy in indicator_configs:
            for col, label, color in indicator_configs[strategy]:
                if col in indicators.columns:
                    ax.plot(indicators.index, indicators[col], label=label, alpha=0.7, linewidth=1, color=color)
        
        elif strategy in ['ADX_Trend_Signal', 'RSI_Signal']:
            ax2 = ax.twinx()
            if strategy == 'ADX_Trend_Signal' and 'ADX' in indicators.columns:
                for col, label, color in [('ADX', 'ADX', 'purple'), ('PLUS_DI', '+DI', 'green'), ('MINUS_DI', '-DI', 'red')]:
                    if col in indicators.columns:
                        ax2.plot(indicators.index, indicators[col], label=label, alpha=0.7, linewidth=1, color=color)
                ax2.axhline(y=25, color='purple', linestyle='--', alpha=0.5, label='ADX Threshold (25)')
                ax2.set_ylabel('ADX/DI Values', fontsize=9)
            elif 'RSI_14' in indicators.columns:
                ax2.plot(indicators.index, indicators['RSI_14'], label='RSI 14', alpha=0.7, linewidth=1, color='purple')
                for y_val, color, label in [(70, 'red', 'Overbought (70)'), (30, 'green', 'Oversold (30)')]:
                    ax2.axhline(y=y_val, color=color, linestyle='--', alpha=0.5, label=label)
                ax2.set_ylabel('RSI Values', fontsize=9)
            ax2.legend(fontsize=7, loc='upper right')
            ax2.set_ylim(0, 100)
        
        return ax2
    
    for idx, ticker in enumerate(included_tickers):
        ax = axes_flat[idx]
        try:
            price_data = quotes[ticker].dropna()
            indicators = technical_indicators[ticker]
            signals = trading_signals[ticker]
            best_strategy = best_strategies.get(ticker, {}).get('strategy', 'SMA_Cross_Signal')
            
            ax.plot(price_data.index, price_data.values, label='Price', color='black', linewidth=1)
            ax2 = plot_indicators(ax, indicators, best_strategy)
            signal_info = plot_signals(ax, signals, price_data, best_strategy, indicators, ax2)
            
            metrics = best_strategies.get(ticker, {})
            title_text = f'{ticker} - {best_strategy}\nCAGR: {metrics.get("cagr", 0):.2%} | Vol: {metrics.get("volatility", 0):.2%} | Sharpe: {metrics.get("sharpe_ratio", 0):.2f} | Max DD: {metrics.get("max_drawdown", 0):.2%} ✓'
            if signal_info:
                title_text += f'\n{signal_info}'
            
            ax.set_title(title_text, fontsize=11, weight='bold')
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting {ticker}\n{str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    for idx in range(len(included_tickers), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance(analysis_results=None):
    """Plot portfolio performance chart using the backtest results"""
    # Accept analysis_results as parameter or use global
    if analysis_results is None:
        if 'analysis_results' not in globals():
            print("Portfolio results not available for plotting")
            return
        analysis_results = globals()['analysis_results']
    
    if 'portfolio_result' not in analysis_results:
        print("Portfolio results not available for plotting")
        return
        
    portfolio_result = analysis_results['portfolio_result']
    fig = plt.figure(figsize=(32, 35))
    gs = fig.add_gridspec(2, 4, height_ratios=[4, 1], width_ratios=[2, 1, 1, 1], hspace=0.5, wspace=0.3)
    ax_main = fig.add_subplot(gs[0, :])
    
    # Main chart
    try:
        portfolio_name = portfolio_result.stats.columns[0]
        portfolio_data = portfolio_result[portfolio_name]
        portfolio_returns = (portfolio_data / portfolio_data.iloc[0] - 1) * 100
        portfolio_returns.plot(ax=ax_main, color='blue', linewidth=2, legend=False)
        ax_main.set_ylabel('Portfolio Return (%)', fontsize=12)
    except:
        try:
            portfolio_result.plot(ax=ax_main, color='blue', linewidth=2, legend=False)
            ax_main.set_ylabel('Portfolio Value', fontsize=12)
        except Exception as e:
            ax_main.text(0.5, 0.5, f'Chart unavailable\nError: {str(e)}', ha='center', va='center', 
                        transform=ax_main.transAxes, fontsize=14, 
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax_main.set_title('Short-Term Portfolio Performance', fontsize=16, weight='bold', pad=20)
    ax_main.set_xlabel('Date', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.tick_params(axis='x', rotation=45)
    
    # Metrics
    try:
        portfolio_stats = analysis_results['portfolio_stats']
        included_tickers = analysis_results.get('included_tickers', [])
        best_strategies = analysis_results.get('best_strategies', {})
        
        # Calculate metrics data
        if included_tickers and best_strategies:
            individual_metrics = {k: [best_strategies[t][k] * (100 if k != 'sharpe_ratio' else 1) for t in included_tickers if t in best_strategies]
                                for k in ['cagr', 'max_drawdown', 'sharpe_ratio', 'volatility']}
            metrics_data = [([min(m), np.mean(m), max(m)] if m else [portfolio_stats[k] * (100 if k != 'sharpe_ratio' else 1)], title, color) 
                           for m, k, title, color in [(individual_metrics['cagr'], 'cagr', 'CAGR', 'green'),
                                                      (individual_metrics['max_drawdown'], 'max_drawdown', 'Drawdown', 'red'),
                                                      (individual_metrics['sharpe_ratio'], 'sharpe_ratio', 'Sharpe Ratio', 'orange'),
                                                      (individual_metrics['volatility'], 'volatility', 'Volatility', 'purple')]]
        else:
            metrics_data = [(portfolio_stats[k] * (100 if k != 'sharpe_ratio' else 1), title, color)
                           for k, title, color in [('cagr', 'CAGR', 'green'), ('max_drawdown', 'Drawdown', 'red'),
                                                  ('sharpe_ratio', 'Sharpe Ratio', 'orange'), ('volatility', 'Volatility', 'purple')]]
        
        for i, (data, title, color) in enumerate(metrics_data):
            create_metrics_subplot(fig, gs, i, data, title, color, i != 2, 2 if i == 2 else 1)
        
        # Add metrics text box
        num_stocks = len(included_tickers)
        avg_metrics = {k: np.mean([best_strategies[t][k] for t in included_tickers if t in best_strategies]) if included_tickers and best_strategies else portfolio_stats[k] 
                      for k in ['sharpe_ratio', 'volatility', 'max_drawdown']}
        
        metrics_text = f"""Portfolio Composition:
• Stocks: {num_stocks}
• Strategy: Technical Signal-Based
• Weight: Equal-weighted ({100/num_stocks:.1f}% each)
• Filters: CAGR ≥10%, Volatility ≤30%

📈 Returns:
• CAGR: {portfolio_stats['cagr']:.2%}
• Total Return: {portfolio_stats['total_return']:.2%}

⚖️ Risk Metrics (Portfolio Avg):
• Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}
• Volatility: {avg_metrics['volatility']:.2%}
• Drawdown: {avg_metrics['max_drawdown']:.2%}"""

        ax_main.text(1.02, 0.98, metrics_text, transform=ax_main.transAxes, va='top', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(right=0.80)
            
    except Exception as e:
        print(f"Could not add metrics: {e}")
        for i, (title, color) in enumerate(zip(['CAGR', 'Max Drawdown', 'Sharpe Ratio', 'Volatility'], 
                                              ['green', 'red', 'orange', 'purple'])):
            ax = fig.add_subplot(gs[1, i])
            ax.text(0.5, 0.5, f'{title}\nData Unavailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    plt.show()

def create_metrics_subplot(fig, gs, position, data, title, color, is_percentage=True, decimal_places=1):
    """Helper function to create consistent metric subplots"""
    ax = fig.add_subplot(gs[1, position])
    
    if isinstance(data, list) and len(data) == 3:
        bp = ax.boxplot([data], patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        ax.text(1, max(data) + max(data)*0.05, f'{data[1]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    else:
        ax.bar(['Portfolio'], [data], color=[color], alpha=0.7, edgecolor='black', linewidth=1)
        label = f'{data:.{decimal_places}f}%' if is_percentage else f'{data:.{decimal_places+1}f}'
        ax.text(0, data + data*0.05, label, ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_ylabel(''), ax.set_xlabel(''), ax.set_xticklabels([])
    ax.grid(True, alpha=0.3, axis='y')

def plot_parameter_comparison(optimization_summary):
    """Plot comparison of optimized parameters across tickers using optimization_summary DataFrame"""
    if optimization_summary.empty:
        print("No optimization summary data to plot")
        return
    
    # Group by strategy type
    strategy_groups = optimization_summary.groupby('Strategy')
    
    for strategy, group in strategy_groups:
        plt.figure(figsize=(15, 10))
        
        if strategy in ['SMA_Cross_Signal', 'EMA_Cross_Signal']:
            # Plot for crossover strategies
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{strategy} - Parameter Optimization Results', fontsize=16, fontweight='bold')
            
            # Subplot 1: Short Period vs Sharpe Ratio
            axes[0,0].bar(group['Ticker'], group['Short_Period'], color='skyblue', alpha=0.7)
            axes[0,0].set_title('Optimal Short Period by Ticker')
            axes[0,0].set_ylabel('Short Period')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Subplot 2: Long Period vs Sharpe Ratio  
            axes[0,1].bar(group['Ticker'], group['Long_Period'], color='lightcoral', alpha=0.7)
            axes[0,1].set_title('Optimal Long Period by Ticker')
            axes[0,1].set_ylabel('Long Period')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Subplot 3: Sharpe Ratio comparison
            axes[1,0].bar(group['Ticker'], group['Best_Sharpe'], color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Best Sharpe Ratio by Ticker')
            axes[1,0].set_ylabel('Sharpe Ratio')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Subplot 4: CAGR vs Volatility scatter
            scatter = axes[1,1].scatter(group['Volatility'], group['CAGR'], 
                                      c=group['Best_Sharpe'], cmap='RdYlGn', 
                                      s=100, alpha=0.7)
            axes[1,1].set_xlabel('Volatility')
            axes[1,1].set_ylabel('CAGR')
            axes[1,1].set_title('Risk-Return Profile')
            plt.colorbar(scatter, ax=axes[1,1], label='Sharpe Ratio')
            
            # Add ticker labels to scatter plot
            for i, ticker in enumerate(group['Ticker']):
                axes[1,1].annotate(ticker, (group['Volatility'].iloc[i], group['CAGR'].iloc[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        elif strategy == 'RSI_Signal':
            # Plot for RSI strategy
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{strategy} - Parameter Optimization Results', fontsize=16, fontweight='bold')
            
            # RSI Period
            axes[0,0].bar(group['Ticker'], group['RSI_Period'], color='gold', alpha=0.7)
            axes[0,0].set_title('Optimal RSI Period by Ticker')
            axes[0,0].set_ylabel('RSI Period')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Oversold Level
            axes[0,1].bar(group['Ticker'], group['Oversold_Level'], color='orange', alpha=0.7)
            axes[0,1].set_title('Optimal Oversold Level by Ticker')
            axes[0,1].set_ylabel('Oversold Level')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Overbought Level
            axes[1,0].bar(group['Ticker'], group['Overbought_Level'], color='red', alpha=0.7)
            axes[1,0].set_title('Optimal Overbought Level by Ticker')
            axes[1,0].set_ylabel('Overbought Level')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Performance metrics
            axes[1,1].bar(group['Ticker'], group['Best_Sharpe'], color='lightgreen', alpha=0.7)
            axes[1,1].set_title('Best Sharpe Ratio by Ticker')
            axes[1,1].set_ylabel('Sharpe Ratio')
            axes[1,1].tick_params(axis='x', rotation=45)
            
        elif strategy == 'ADX_Trend_Signal':
            # Plot for ADX strategy
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{strategy} - Parameter Optimization Results', fontsize=16, fontweight='bold')
            
            # ADX Period
            axes[0,0].bar(group['Ticker'], group['ADX_Period'], color='purple', alpha=0.7)
            axes[0,0].set_title('Optimal ADX Period by Ticker')
            axes[0,0].set_ylabel('ADX Period')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # ADX Threshold
            axes[0,1].bar(group['Ticker'], group['ADX_Threshold'], color='indigo', alpha=0.7)
            axes[0,1].set_title('Optimal ADX Threshold by Ticker')
            axes[0,1].set_ylabel('ADX Threshold')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Performance metrics
            axes[1,0].bar(group['Ticker'], group['Best_Sharpe'], color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Best Sharpe Ratio by Ticker')
            axes[1,0].set_ylabel('Sharpe Ratio')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Risk-return scatter
            scatter = axes[1,1].scatter(group['Volatility'], group['CAGR'], 
                                      c=group['Best_Sharpe'], cmap='RdYlGn', 
                                      s=100, alpha=0.7)
            axes[1,1].set_xlabel('Volatility')
            axes[1,1].set_ylabel('CAGR')
            axes[1,1].set_title('Risk-Return Profile')
            plt.colorbar(scatter, ax=axes[1,1], label='Sharpe Ratio')
            
            # Add ticker labels
            for i, ticker in enumerate(group['Ticker']):
                axes[1,1].annotate(ticker, (group['Volatility'].iloc[i], group['CAGR'].iloc[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.show()

def plot_portfolio_performance(optimization_summary):
    """Plot portfolio performance using optimization_summary DataFrame"""
    if optimization_summary is None or optimization_summary.empty:
        print("No optimization summary data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sharpe Ratio by Ticker
    tickers = optimization_summary['Ticker']
    sharpe_ratios = optimization_summary['Best_Sharpe']
    colors = ['green' if x > 1.5 else 'orange' if x > 1.0 else 'red' for x in sharpe_ratios]
    
    axes[0,0].bar(tickers, sharpe_ratios, color=colors, alpha=0.7)
    axes[0,0].set_title('Sharpe Ratio by Ticker')
    axes[0,0].set_ylabel('Sharpe Ratio')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Threshold')
    axes[0,0].legend()
    
    # Plot 2: CAGR vs Volatility
    scatter = axes[0,1].scatter(optimization_summary['Volatility'] * 100, 
                               optimization_summary['CAGR'] * 100,
                               c=optimization_summary['Best_Sharpe'], 
                               cmap='RdYlGn', s=100, alpha=0.7)
    axes[0,1].set_xlabel('Volatility (%)')
    axes[0,1].set_ylabel('CAGR (%)')
    axes[0,1].set_title('Risk-Return Profile')
    plt.colorbar(scatter, ax=axes[0,1], label='Sharpe Ratio')
    
    # Add ticker labels
    for i, ticker in enumerate(optimization_summary['Ticker']):
        axes[0,1].annotate(ticker, 
                          (optimization_summary['Volatility'].iloc[i] * 100, 
                           optimization_summary['CAGR'].iloc[i] * 100),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Strategy Distribution
    strategy_counts = optimization_summary['Strategy'].value_counts()
    axes[1,0].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Strategy Distribution')
    
    # Plot 4: Max Drawdown
    axes[1,1].bar(tickers, optimization_summary['Max_Drawdown'] * 100, 
                  color='red', alpha=0.7)
    axes[1,1].set_title('Maximum Drawdown by Ticker')
    axes[1,1].set_ylabel('Max Drawdown (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_ticker_signals_with_annotations(quotes, trading_signals, technical_indicators, optimization_summary):
    """Plot trading signals for tickers using optimization_summary DataFrame"""
    if optimization_summary is None or optimization_summary.empty:
        print("No optimization summary data to plot")
        return
    
    tickers_to_plot = optimization_summary['Ticker'].tolist()[:6]  # Plot first 6 tickers
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Trading Signals Analysis', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, ticker in enumerate(tickers_to_plot):
        if ticker.lower() not in quotes.columns:
            print(f"Ticker {ticker} not found in quotes data")
            continue
            
        # Get ticker data
        price_data = quotes[ticker.lower()]
        ticker_row = optimization_summary[optimization_summary['Ticker'] == ticker].iloc[0]
        strategy = ticker_row['Strategy']
        
        # Plot price
        axes[i].plot(price_data.index, price_data.values, label='Price', linewidth=1)
        
        # Add strategy-specific indicators and signals
        if strategy in ['SMA_Cross_Signal', 'EMA_Cross_Signal'] and ticker.lower() in technical_indicators:
            indicators = technical_indicators[ticker.lower()]
            
            if 'SMA_10' in indicators.columns:
                axes[i].plot(indicators.index, indicators['SMA_10'], 
                           label=f'SMA {int(ticker_row["Short_Period"])}', alpha=0.7)
            if 'SMA_25' in indicators.columns:
                axes[i].plot(indicators.index, indicators['SMA_25'], 
                           label=f'SMA {int(ticker_row["Long_Period"])}', alpha=0.7)
        
        # Add performance metrics as text
        axes[i].text(0.02, 0.98, f'Sharpe: {ticker_row["Best_Sharpe"]:.2f}\nCAGR: {ticker_row["CAGR"]*100:.1f}%', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[i].set_title(f'{ticker} - {strategy}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
