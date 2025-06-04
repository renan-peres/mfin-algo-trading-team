import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import talib
import bt
from itertools import product
import warnings
import logging
import os
import sys
from io import StringIO

import matplotlib.pyplot as plt

# Suppress warnings and logging
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

def calculate_technical_indicators(prices_df, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """
    Calculate SMA and EMA technical indicators with customizable timeperiods
    
    Parameters:
    - prices_df: DataFrame with stock prices
    - sma_short: Short-term SMA period (default: 50)
    - sma_long: Long-term SMA period (default: 200)
    - ema_short: Short-term EMA period (default: 26)
    - ema_long: Long-term EMA period (default: 52)
    
    Returns a dictionary of DataFrames with indicators for each stock
    """
    indicators = {}
    
    for ticker in prices_df.columns:       
        close = prices_df[ticker].dropna()
        min_required = max(sma_long, ema_long) + 10  # Extra buffer for stability
        
        if len(close) < min_required:  # Skip if insufficient data
            print(f"  ⚠️ Skipping {ticker}: insufficient data ({len(close)} points, need {min_required})")
            continue
        
        ticker_indicators = pd.DataFrame(index=close.index)
        ticker_indicators['Close'] = close
        
        try:
            # Simple Moving Averages with custom periods
            ticker_indicators[f'SMA_{sma_short}'] = talib.SMA(close, timeperiod=sma_short)
            ticker_indicators[f'SMA_{sma_long}'] = talib.SMA(close, timeperiod=sma_long)
            
            # Exponential Moving Averages with custom periods
            ticker_indicators[f'EMA_{ema_short}'] = talib.EMA(close, timeperiod=ema_short)
            ticker_indicators[f'EMA_{ema_long}'] = talib.EMA(close, timeperiod=ema_long)
            
            indicators[ticker] = ticker_indicators
            
        except Exception as e:
            print(f"  ❌ Error calculating indicators for {ticker}: {str(e)}")
            continue
    
    print(f"✅ Technical indicators calculated for {len(indicators)} tickers")
    print(f"   Using periods: SMA({sma_short}, {sma_long}), EMA({ema_short}, {ema_long})")
    return indicators

def generate_trading_signals(indicators_dict, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """
    Generate buy/sell signals based on SMA and EMA crossovers with customizable periods
    
    Parameters:
    - indicators_dict: Dictionary of indicator DataFrames
    - sma_short, sma_long, ema_short, ema_long: Periods matching those used in calculate_technical_indicators
    
    Returns a dictionary of signal DataFrames for each strategy
    """
    strategies = {}
    
    sma_short_col = f'SMA_{sma_short}'
    sma_long_col = f'SMA_{sma_long}'
    ema_short_col = f'EMA_{ema_short}'
    ema_long_col = f'EMA_{ema_long}'
    
    for ticker, indicators in indicators_dict.items():
        try:
            ticker_signals = pd.DataFrame(index=indicators.index)
            
            # Strategy 1: SMA Crossover
            ticker_signals['SMA_Cross_Signal'] = np.where(
                indicators[sma_short_col] > indicators[sma_long_col], 1, 
                np.where(indicators[sma_short_col] < indicators[sma_long_col], -1, 0)
            )
            
            # Strategy 2: EMA Crossover
            ticker_signals['EMA_Cross_Signal'] = np.where(
                indicators[ema_short_col] > indicators[ema_long_col], 1, 
                np.where(indicators[ema_short_col] < indicators[ema_long_col], -1, 0)
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
    Optimize parameters for SMA and EMA crossover strategies based on best performing strategies.
    Includes expanded parameter ranges to cover original baseline parameters.
    """
    optimization_results = {}
    
    # Expanded parameter ranges including original baseline parameters
    parameter_ranges = {
        'SMA_Cross_Signal': {
            'short_periods': list(range(10, 60, 5)) + [50],   # 10,15,20...55 + original 50
            'long_periods': list(range(50, 250, 10)) + [200]  # 50,60,70...240 + original 200
        },
        'EMA_Cross_Signal': {
            'short_periods': list(range(8, 35, 2)) + [26],    # 8,10,12...34 + original 26
            'long_periods': list(range(30, 70, 5)) + [52]     # 30,35,40...65 + original 52
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
            if len(ticker_data) < 250:  # Increased minimum for longer periods
                print(f"  ⚠️ Insufficient data for {ticker} (need ≥250 points for long periods)")
                continue
                
            if best_strategy == 'SMA_Cross_Signal':
                results = optimize_sma_parameters(ticker, ticker_data, parameter_ranges['SMA_Cross_Signal'], risk_free_rate)
            elif best_strategy == 'EMA_Cross_Signal':
                results = optimize_ema_parameters(ticker, ticker_data, parameter_ranges['EMA_Cross_Signal'], risk_free_rate)
            else:
                print(f"  ⚠️ Strategy {best_strategy} not supported (only SMA/EMA crossover strategies)")
                continue
                
            optimization_results[ticker] = results
            
        except Exception as e:
            print(f"  ❌ Error optimizing {ticker}: {str(e)}")
            continue
    
    return optimization_results

def optimize_sma_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize SMA crossover parameters with comprehensive parameter sweep."""
    results = []
    total_combinations = len(param_ranges['short_periods']) * len(param_ranges['long_periods'])
    print(f"  Testing {total_combinations} parameter combinations...")
    
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
            result = run_backtest_silent(backtest)
            
            # Extract metrics using safe extraction
            stats = result.stats
            cagr = safe_get_stat(stats, strategy_name, ['cagr'])
            sharpe = safe_get_stat(stats, strategy_name, ['daily_sharpe', 'monthly_sharpe', 'yearly_sharpe'])
            max_dd = safe_get_stat(stats, strategy_name, ['max_drawdown'])
            volatility = safe_get_stat(stats, strategy_name, ['daily_vol', 'monthly_vol', 'yearly_vol'])
            
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
        # Filter out invalid results
        valid_results = results_df[
            (results_df['sharpe_ratio'] != 0) & 
            (~results_df['sharpe_ratio'].isna()) & 
            (~results_df['sharpe_ratio'].isin([np.inf, -np.inf]))
        ]
        
        if not valid_results.empty:
            best_params = valid_results.loc[valid_results['sharpe_ratio'].idxmax()]
            print(f"  ✅ Best SMA params for {ticker}: {best_params['short_period']}/{best_params['long_period']} "
                  f"(Sharpe: {best_params['sharpe_ratio']:.3f}, CAGR: {best_params['cagr']:.2%})")
            return {
                'results_df': results_df, 
                'best_params': best_params, 
                'strategy_type': 'SMA_Cross_Signal',
                'valid_combinations': len(valid_results)
            }
        else:
            print(f"  ⚠️ No valid parameter combinations found for {ticker}")
    
    return None

def optimize_ema_parameters(ticker, data, param_ranges, risk_free_rate):
    """Optimize EMA crossover parameters with comprehensive parameter sweep."""
    results = []
    total_combinations = len(param_ranges['short_periods']) * len(param_ranges['long_periods'])
    print(f"  Testing {total_combinations} parameter combinations...")
    
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
            result = run_backtest_silent(backtest)
            
            # Extract metrics using safe extraction
            stats = result.stats
            cagr = safe_get_stat(stats, strategy_name, ['cagr'])
            sharpe = safe_get_stat(stats, strategy_name, ['daily_sharpe', 'monthly_sharpe', 'yearly_sharpe'])
            max_dd = safe_get_stat(stats, strategy_name, ['max_drawdown'])
            volatility = safe_get_stat(stats, strategy_name, ['daily_vol', 'monthly_vol', 'yearly_vol'])
            
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
        # Filter out invalid results
        valid_results = results_df[
            (results_df['sharpe_ratio'] != 0) & 
            (~results_df['sharpe_ratio'].isna()) & 
            (~results_df['sharpe_ratio'].isin([np.inf, -np.inf]))
        ]
        
        if not valid_results.empty:
            best_params = valid_results.loc[valid_results['sharpe_ratio'].idxmax()]
            print(f"  ✅ Best EMA params for {ticker}: {best_params['short_period']}/{best_params['long_period']} "
                  f"(Sharpe: {best_params['sharpe_ratio']:.3f}, CAGR: {best_params['cagr']:.2%})")
            return {
                'results_df': results_df, 
                'best_params': best_params, 
                'strategy_type': 'EMA_Cross_Signal',
                'valid_combinations': len(valid_results)
            }
        else:
            print(f"  ⚠️ No valid parameter combinations found for {ticker}")
    
    return None

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

def plot_optimization_heatmaps(optimization_results):
    """Plot heatmaps only for the selected optimal tickers"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get the selected tickers and their optimization data
    selected_tickers = optimization_results['selected_tickers']
    all_optimization_results = optimization_results['all_optimization_results']
    
    print(f"📊 Creating heatmaps for {len(selected_tickers)} selected tickers...")
    
    for ticker in selected_tickers:
        if ticker not in all_optimization_results or all_optimization_results[ticker] is None:
            print(f"⚠️ No optimization data for {ticker} - skipping")
            continue
        
        results = all_optimization_results[ticker]
        
        # Check if detailed results are available
        if 'results_df' not in results:
            print(f"⚠️ No detailed results data for {ticker} - skipping heatmap")
            continue
            
        results_df = results['results_df']
        strategy_type = results['strategy_type']
        best_params = results['best_params']
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for crossover strategies
        pivot_table = results_df.pivot_table(
            values='sharpe_ratio', 
            index='long_period', 
            columns='short_period', 
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Sharpe Ratio'})
        
        # Add title with best parameters
        plt.title(f'{ticker} - {strategy_type} Parameter Optimization\n'
                 f'Best: {best_params["short_period"]}/{best_params["long_period"]} '
                 f'(Sharpe: {best_params["sharpe_ratio"]:.3f})')
        plt.xlabel('Short Period')
        plt.ylabel('Long Period')
        
        # Highlight the best parameter combination
        best_short = best_params['short_period']
        best_long = best_params['long_period']
        
        # Find the position in the heatmap
        if best_short in pivot_table.columns and best_long in pivot_table.index:
            short_idx = list(pivot_table.columns).index(best_short)
            long_idx = list(pivot_table.index).index(best_long)
            
            # Add red border around best cell
            plt.gca().add_patch(plt.Rectangle((short_idx, long_idx), 1, 1, 
                                            fill=False, edgecolor='red', lw=3))
        
        plt.tight_layout()
        plt.show()

def optimize_single_ticker_parameters(args):
    """
    Optimized single ticker parameter optimization with vectorized operations
    """
    ticker, ticker_data, parameter_ranges = args
    
    print(f"🔄 Optimizing {ticker}...")
    
    if len(ticker_data) < 250:
        return ticker, None
    
    close_prices = ticker_data[ticker].values
    dates = ticker_data.index
    
    all_results = []
    
    # Pre-compute all combinations to avoid nested loops
    sma_combinations = list(product(parameter_ranges['SMA_Cross_Signal']['short_periods'], 
                                   parameter_ranges['SMA_Cross_Signal']['long_periods']))
    ema_combinations = list(product(parameter_ranges['EMA_Cross_Signal']['short_periods'], 
                                   parameter_ranges['EMA_Cross_Signal']['long_periods']))
    
    # Vectorized SMA optimization
    for short_period, long_period in sma_combinations:
        if short_period >= long_period:
            continue
            
        try:
            # Use numpy for faster calculations
            sma_short = pd.Series(close_prices).rolling(short_period).mean().values
            sma_long = pd.Series(close_prices).rolling(long_period).mean().values
            
            # Vectorized signal generation
            signals = np.where(sma_short > sma_long, 1.0, 
                             np.where(sma_short < sma_long, -1.0, 0.0))
            signals = np.nan_to_num(signals, 0.0)
            
            # Quick performance calculation using numpy
            returns = np.diff(close_prices) / close_prices[:-1]
            strategy_returns = returns * signals[:-1]
            
            # Fast metrics calculation
            if len(strategy_returns) > 0 and np.any(strategy_returns != 0):
                total_return = np.prod(1 + strategy_returns) - 1
                volatility = np.std(strategy_returns) * np.sqrt(252)
                sharpe = (np.mean(strategy_returns) * 252) / (volatility + 1e-10)
                
                # Simple drawdown calculation
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative / running_max) - 1
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Annualized return
                years = len(strategy_returns) / 252
                cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                
                all_results.append({
                    'strategy_type': 'SMA_Cross_Signal',
                    'short_period': short_period,
                    'long_period': long_period,
                    'cagr': cagr,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'total_return': total_return
                })
        except:
            continue
    
    # Vectorized EMA optimization
    for short_period, long_period in ema_combinations:
        if short_period >= long_period:
            continue
            
        try:
            # Use pandas EWM for EMA calculation
            close_series = pd.Series(close_prices)
            ema_short = close_series.ewm(span=short_period).mean().values
            ema_long = close_series.ewm(span=long_period).mean().values
            
            # Vectorized signal generation
            signals = np.where(ema_short > ema_long, 1.0,
                             np.where(ema_short < ema_long, -1.0, 0.0))
            signals = np.nan_to_num(signals, 0.0)
            
            # Quick performance calculation
            returns = np.diff(close_prices) / close_prices[:-1]
            strategy_returns = returns * signals[:-1]
            
            # Fast metrics calculation
            if len(strategy_returns) > 0 and np.any(strategy_returns != 0):
                total_return = np.prod(1 + strategy_returns) - 1
                volatility = np.std(strategy_returns) * np.sqrt(252)
                sharpe = (np.mean(strategy_returns) * 252) / (volatility + 1e-10)
                
                # Simple drawdown calculation
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative / running_max) - 1
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Annualized return
                years = len(strategy_returns) / 252
                cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                
                all_results.append({
                    'strategy_type': 'EMA_Cross_Signal',
                    'short_period': short_period,
                    'long_period': long_period,
                    'cagr': cagr,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'total_return': total_return
                })
        except:
            continue
    
    if not all_results:
        return ticker, None
    
    # Find best strategy based on Sharpe ratio
    results_df = pd.DataFrame(all_results)
    valid_results = results_df[
        (results_df['sharpe_ratio'] != 0) & 
        (~results_df['sharpe_ratio'].isna()) & 
        (~results_df['sharpe_ratio'].isin([np.inf, -np.inf]))
    ]
    
    if valid_results.empty:
        return ticker, None
    
    best_idx = valid_results['sharpe_ratio'].idxmax()
    best_params = valid_results.loc[best_idx].to_dict()
    
    return ticker, {
        'best_strategy': best_params['strategy_type'],
        'best_params': best_params,
        'strategy_type': best_params['strategy_type'],
        'results_df': results_df
    }

def find_optimal_portfolio_with_parameter_optimization(quotes, min_cagr=0.0, max_volatility=0.3, max_stocks=10, n_jobs=None):
    """
    Ultra-fast optimized version using parallel processing and vectorized operations
    """
    print("=" * 80)
    print("FAST PORTFOLIO OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, len(quotes.columns))
    
    print(f"🚀 Using {n_jobs} parallel processes")
    
    available_tickers = list(quotes.columns)
    
    # Reduced parameter ranges for speed while maintaining effectiveness
    parameter_ranges = {
        'SMA_Cross_Signal': {
            'short_periods': list(range(10, 60, 10)) + [20, 50],  # Reduced granularity
            'long_periods': list(range(50, 200, 20)) + [100, 200]  # Reduced granularity
        },
        'EMA_Cross_Signal': {
            'short_periods': list(range(8, 35, 4)) + [12, 26],    # Reduced granularity
            'long_periods': list(range(30, 65, 8)) + [50, 52]     # Reduced granularity
        }
    }
    
    # Prepare data for parallel processing
    print(f"\n🔄 Phase 1: Parallel Parameter Optimization for {len(available_tickers)} tickers")
    
    # Filter tickers with sufficient data upfront
    valid_tickers = []
    ticker_data_list = []
    
    for ticker in available_tickers:
        ticker_data = quotes[[ticker]].dropna()
        if len(ticker_data) >= 250:
            valid_tickers.append(ticker)
            ticker_data_list.append((ticker, ticker_data, parameter_ranges))
    
    print(f"📊 Processing {len(valid_tickers)} tickers with sufficient data")
    
    # Parallel optimization
    all_optimization_results = {}
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(optimize_single_ticker_parameters, args): args[0] 
            for args in ticker_data_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_ticker):
            ticker, result = future.result()
            if result is not None:
                all_optimization_results[ticker] = result
                print(f"✅ {ticker} completed ({completed+1}/{len(ticker_data_list)})")
            else:
                print(f"⚠️ {ticker} failed optimization")
            completed += 1
    
    print(f"\n📊 Optimization completed for {len(all_optimization_results)} tickers")
    
    if not all_optimization_results:
        print("❌ No successful optimizations!")
        return None
    
    # Step 2: Fast filtering using vectorized operations
    print("\n🔄 Phase 2: Fast Performance Filtering")
    
    # Create DataFrame for vectorized filtering
    filter_data = []
    for ticker, results in all_optimization_results.items():
        params = results['best_params']
        filter_data.append({
            'ticker': ticker,
            'cagr': params['cagr'],
            'volatility': params['volatility'],
            'sharpe_ratio': params['sharpe_ratio']
        })
    
    filter_df = pd.DataFrame(filter_data)
    
    # Vectorized filtering
    qualified_mask = (filter_df['cagr'] >= min_cagr) & (filter_df['volatility'] <= max_volatility)
    qualified_df = filter_df[qualified_mask]
    
    print(f"📊 Filtering Results:")
    print(f"  • Total optimized: {len(filter_df)}")
    print(f"  • Qualified (CAGR ≥{min_cagr:.1%}, Vol ≤{max_volatility:.1%}): {len(qualified_df)}")
    
    if qualified_df.empty:
        print("⚠️ No tickers meet performance criteria!")
        return None
    
    # Step 3: Select top performers
    top_df = qualified_df.nlargest(max_stocks, 'sharpe_ratio')
    top_tickers = top_df['ticker'].tolist()
    
    print(f"\n🏆 Selected top {len(top_tickers)} performers:")
    for i, ticker in enumerate(top_tickers, 1):
        row = top_df[top_df['ticker'] == ticker].iloc[0]
        params = all_optimization_results[ticker]['best_params']
        strategy = all_optimization_results[ticker]['strategy_type']
        print(f"  #{i}. {ticker}: {strategy} ({params['short_period']}/{params['long_period']}) - "
              f"Sharpe: {row['sharpe_ratio']:.3f}, CAGR: {row['cagr']:.2%}")
    
    # Step 4: Fast signal generation using vectorized operations
    print("\n🔄 Phase 3: Fast Signal Generation")
    
    optimized_signals = {}
    portfolio_weights = pd.DataFrame(index=quotes.index, columns=quotes.columns).fillna(0)
    weight_per_ticker = 1.0 / len(top_tickers)
    
    for ticker in top_tickers:
        params = all_optimization_results[ticker]['best_params']
        strategy_type = params['strategy_type']
        
        # Fast signal calculation
        close = quotes[ticker].dropna()
        
        if strategy_type == 'SMA_Cross_Signal':
            # Vectorized SMA calculation
            sma_short = close.rolling(params['short_period']).mean()
            sma_long = close.rolling(params['long_period']).mean()
            signals = np.where(sma_short > sma_long, 1, 
                             np.where(sma_short < sma_long, -1, 0))
        else:  # EMA_Cross_Signal
            # Vectorized EMA calculation
            ema_short = close.ewm(span=params['short_period']).mean()
            ema_long = close.ewm(span=params['long_period']).mean()
            signals = np.where(ema_short > ema_long, 1,
                             np.where(ema_short < ema_long, -1, 0))
        
        # Create signals DataFrame
        ticker_signals = pd.DataFrame(index=close.index)
        ticker_signals[strategy_type] = signals
        optimized_signals[ticker] = ticker_signals
        
        # Add to portfolio weights
        reindexed_signals = pd.Series(signals, index=close.index).reindex(quotes.index, method='ffill').fillna(0)
        portfolio_weights[ticker] = reindexed_signals * weight_per_ticker
    
    # Step 5: Fast portfolio backtesting
    print("\n🔄 Phase 4: Fast Portfolio Backtesting")
    
    # Simple portfolio performance calculation
    portfolio_returns = []
    for date in quotes.index[1:]:
        daily_return = 0
        for ticker in top_tickers:
            if ticker in quotes.columns and date in quotes.index:
                try:
                    price_return = quotes[ticker].loc[date] / quotes[ticker].shift(1).loc[date] - 1
                    weight = portfolio_weights[ticker].loc[date]
                    daily_return += price_return * weight
                except:
                    continue
        portfolio_returns.append(daily_return)
    
    # Calculate portfolio statistics
    portfolio_returns = np.array(portfolio_returns)
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    
    if len(portfolio_returns) > 0:
        portfolio_total_return = np.prod(1 + portfolio_returns) - 1
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_sharpe = (np.mean(portfolio_returns) * 252) / (portfolio_volatility + 1e-10)
        
        # Portfolio drawdown
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1
        portfolio_max_drawdown = np.min(portfolio_drawdown)
        
        # Portfolio CAGR
        years = len(portfolio_returns) / 252
        portfolio_cagr = (1 + portfolio_total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        portfolio_cagr = portfolio_sharpe = portfolio_volatility = portfolio_max_drawdown = 0
    
    portfolio_stats = {
        'cagr': portfolio_cagr,
        'total_return': portfolio_total_return if 'portfolio_total_return' in locals() else 0,
        'sharpe_ratio': portfolio_sharpe,
        'volatility': portfolio_volatility,
        'max_drawdown': portfolio_max_drawdown
    }
    
    # Step 6: Create summary results
    optimization_summary = []
    best_strategies_dict = {}
    
    for ticker in top_tickers:
        params = all_optimization_results[ticker]['best_params']
        
        optimization_summary.append({
            'Ticker': ticker,
            'Strategy': params['strategy_type'],
            'Best_Sharpe': params['sharpe_ratio'],
            'CAGR': params['cagr'],
            'Max_Drawdown': params['max_drawdown'],
            'Volatility': params['volatility'],
            'Short_Period': params['short_period'],
            'Long_Period': params['long_period']
        })
        
        best_strategies_dict[ticker] = {
            'strategy': params['strategy_type'],
            'cagr': params['cagr'],
            'total_return': params.get('total_return', 0),
            'max_drawdown': params['max_drawdown'],
            'sharpe_ratio': params['sharpe_ratio'],
            'volatility': params['volatility']
        }
    
    optimization_summary_df = pd.DataFrame(optimization_summary)
    best_strategies_df = pd.DataFrame.from_dict(best_strategies_dict, orient='index')
    
    # Display results
    print("✅ Fast Optimization Complete!")
    print(f"\n📈 Optimized Portfolio: {len(top_tickers)} stocks")
    print(f"  📊 Portfolio CAGR: {portfolio_stats['cagr']:.2%}")
    print(f"  📊 Portfolio Sharpe: {portfolio_stats['sharpe_ratio']:.3f}")
    print(f"  📊 Portfolio Volatility: {portfolio_stats['volatility']:.2%}")
    print(f"  📊 Portfolio Max Drawdown: {portfolio_stats['max_drawdown']:.2%}")
    
    avg_sharpe = optimization_summary_df['Best_Sharpe'].mean()
    avg_cagr = optimization_summary_df['CAGR'].mean()
    strategy_distribution = optimization_summary_df['Strategy'].value_counts()
    
    print(f"\n📊 Individual Stock Averages:")
    print(f"  📊 Avg Individual Sharpe: {avg_sharpe:.3f}")
    print(f"  📊 Avg Individual CAGR: {avg_cagr:.2%}")
    print(f"  🎯 Strategy Distribution:")
    for strategy, count in strategy_distribution.items():
        percentage = count / len(optimization_summary_df) * 100
        print(f"    - {strategy}: {count} stocks ({percentage:.1f}%)")
    
    return {
        'portfolio_stats': portfolio_stats,
        'portfolio_weights': portfolio_weights,
        'optimization_summary': optimization_summary_df,
        'best_strategies_df': best_strategies_df,
        'selected_tickers': top_tickers,
        'optimized_signals': optimized_signals,
        'all_optimization_results': all_optimization_results,
        'processing_time': 'Fast parallel processing completed'
    }

# ===============================================================================
# PLOTTING
# ===============================================================================

def plot_complete_strategy_analysis(optimization_results, quotes, training_set, test_set):
    """
    Complete function to generate and plot optimized trading signals for full dataset.
    
    Parameters:
    -----------
    optimization_results : dict
        Results from optimization containing selected tickers and parameters
    quotes : pd.DataFrame
        Full dataset (training + test) with price data
    training_set : pd.DataFrame
        Training period data for background coloring
    test_set : pd.DataFrame
        Test period data for background coloring
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("\n🔄 Generating optimized signals for full dataset (training + test)...")
    
    # Get selected tickers from optimization results
    selected_tickers = optimization_results['selected_tickers']
    optimization_summary = optimization_results['optimization_summary']
    print(f"Selected tickers: {selected_tickers}")
    
    # Create optimized indicators and signals for FULL dataset (training + test)
    full_technical_indicators = {}
    full_trading_signals = {}
    
    for ticker in selected_tickers:
        if ticker in optimization_results['all_optimization_results']:
            params = optimization_results['all_optimization_results'][ticker]['best_params']
            strategy_type = params['strategy_type']
            
            # Get FULL dataset for this ticker (training + test)
            ticker_full_data = quotes[[ticker]].dropna()
            
            # Calculate optimized indicators for full dataset
            ticker_indicators = pd.DataFrame(index=ticker_full_data.index)
            ticker_indicators['Close'] = ticker_full_data[ticker]
            
            if strategy_type == 'SMA_Cross_Signal':
                short_period = params['short_period']
                long_period = params['long_period']
                ticker_indicators[f'SMA_{short_period}'] = ticker_full_data[ticker].rolling(short_period).mean()
                ticker_indicators[f'SMA_{long_period}'] = ticker_full_data[ticker].rolling(long_period).mean()
                
                # Generate improved crossover signals for full dataset
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                
                # Calculate position based on MA relationship
                position = np.where(
                    ticker_indicators[f'SMA_{short_period}'] > ticker_indicators[f'SMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'SMA_{short_period}'] < ticker_indicators[f'SMA_{long_period}'], -1, 0)
                )
                ticker_signals['SMA_Cross_Signal'] = position
                
                # Detect actual crossover points (signal changes)
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                
                # Create buy/sell signals only at crossover points
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)  # 0 to 1 or -1 to 1
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)  # 1 to -1 or 1 to 0
                
                # Also detect entry from neutral
                ticker_signals['Buy_Signal'] = np.where(
                    (signal_changes == 1) & (position_series == 1), 1, ticker_signals['Buy_Signal']
                )
                ticker_signals['Sell_Signal'] = np.where(
                    (signal_changes == -1) & (position_series == -1), 1, ticker_signals['Sell_Signal']
                )
                
            else:  # EMA_Cross_Signal
                short_period = params['short_period']
                long_period = params['long_period']
                ticker_indicators[f'EMA_{short_period}'] = ticker_full_data[ticker].ewm(span=short_period).mean()
                ticker_indicators[f'EMA_{long_period}'] = ticker_full_data[ticker].ewm(span=long_period).mean()
                
                # Generate improved crossover signals for full dataset
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                
                # Calculate position based on MA relationship
                position = np.where(
                    ticker_indicators[f'EMA_{short_period}'] > ticker_indicators[f'EMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'EMA_{short_period}'] < ticker_indicators[f'EMA_{long_period}'], -1, 0)
                )
                ticker_signals['EMA_Cross_Signal'] = position
                
                # Detect actual crossover points (signal changes)
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                
                # Create buy/sell signals only at crossover points
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)  # 0 to 1 or -1 to 1
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)  # 1 to -1 or 1 to 0
                
                # Also detect entry from neutral
                ticker_signals['Buy_Signal'] = np.where(
                    (signal_changes == 1) & (position_series == 1), 1, ticker_signals['Buy_Signal']
                )
                ticker_signals['Sell_Signal'] = np.where(
                    (signal_changes == -1) & (position_series == -1), 1, ticker_signals['Sell_Signal']
                )
            
            full_technical_indicators[ticker] = ticker_indicators
            full_trading_signals[ticker] = ticker_signals
    
    print(f"✅ Generated optimized signals for {len(full_technical_indicators)} tickers")
    
    # Plot the results
    def plot_ticker_signals_with_annotations(quotes, trading_signals, technical_indicators, best_strategies, included_tickers):
        """Plot individual ticker charts with signals and technical indicators for included tickers only"""
        if not included_tickers:
            print("❌ No tickers available for plotting (none passed the CAGR and volatility thresholds)")
            return
        
        print(f"📊 Plotting {len(included_tickers)} tickers that passed the thresholds")
        
        cols = min(2, len(included_tickers))
        rows = (len(included_tickers) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        axes_flat = [axes] if len(included_tickers) == 1 else axes.flatten()
        
        def plot_signals_and_indicators(ax, ticker, indicators, signals, strategy_info):
            """Combined function to plot both indicators and signals"""
            ax2 = None
            strategy = strategy_info.get('Strategy', 'SMA_Cross_Signal')
            
            # Plot moving averages based on optimized parameters
            short_period = strategy_info.get('Short_Period', 20)
            long_period = strategy_info.get('Long_Period', 50)
            
            if strategy == 'SMA_Cross_Signal':
                sma_short_col = f'SMA_{short_period}'
                sma_long_col = f'SMA_{long_period}'
                
                if sma_short_col in indicators.columns and sma_long_col in indicators.columns:
                    ax.plot(indicators.index, indicators[sma_short_col], 
                           label=f'SMA {short_period}', alpha=0.8, color='blue', linewidth=1.5)
                    ax.plot(indicators.index, indicators[sma_long_col], 
                           label=f'SMA {long_period}', alpha=0.8, color='orange', linewidth=1.5)
            
            elif strategy == 'EMA_Cross_Signal':
                ema_short_col = f'EMA_{short_period}'
                ema_long_col = f'EMA_{long_period}'
                
                if ema_short_col in indicators.columns and ema_long_col in indicators.columns:
                    ax.plot(indicators.index, indicators[ema_short_col], 
                           label=f'EMA {short_period}', alpha=0.8, color='green', linewidth=1.5)
                    ax.plot(indicators.index, indicators[ema_long_col], 
                           label=f'EMA {long_period}', alpha=0.8, color='red', linewidth=1.5)
            
            # Plot buy/sell signals at crossover points
            if strategy in signals.columns:
                # Look for Buy_Signal and Sell_Signal columns if available
                buy_signals = signals.get('Buy_Signal', pd.Series(dtype=float, index=signals.index))
                sell_signals = signals.get('Sell_Signal', pd.Series(dtype=float, index=signals.index))
                
                buy_count = sell_count = 0
                
                # Plot buy signals at crossover points
                buy_dates = buy_signals[buy_signals == 1].index
                for date in buy_dates:
                    if date in indicators.index:
                        try:
                            # Use the moving average value at crossover point instead of price
                            if strategy == 'SMA_Cross_Signal':
                                y_val = indicators.loc[date, f'SMA_{short_period}']
                            elif strategy == 'EMA_Cross_Signal':
                                y_val = indicators.loc[date, f'EMA_{short_period}']
                            else:
                                y_val = quotes[ticker].loc[date]  # Fallback to price
                            
                            ax.scatter(date, y_val, color='green', marker='^', s=120, alpha=0.9, zorder=6)
                            ax.annotate('Long', xy=(date, y_val), xytext=(10, 25), 
                                       textcoords='offset points', fontsize=9, fontweight='bold', 
                                       color='green', ha='left', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                                edgecolor='green', alpha=0.9),
                                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.8))
                            buy_count += 1
                        except (KeyError, IndexError):
                            continue
                
                # Plot sell signals at crossover points
                sell_dates = sell_signals[sell_signals == 1].index
                for date in sell_dates:
                    if date in indicators.index:
                        try:
                            # Use the moving average value at crossover point instead of price
                            if strategy == 'SMA_Cross_Signal':
                                y_val = indicators.loc[date, f'SMA_{short_period}']
                            elif strategy == 'EMA_Cross_Signal':
                                y_val = indicators.loc[date, f'EMA_{short_period}']
                            else:
                                y_val = quotes[ticker].loc[date]  # Fallback to price
                            
                            ax.scatter(date, y_val, color='red', marker='v', s=120, alpha=0.9, zorder=6)
                            ax.annotate('Short', xy=(date, y_val), xytext=(10, -25), 
                                       textcoords='offset points', fontsize=9, fontweight='bold', 
                                       color='red', ha='left', va='top',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                                edgecolor='red', alpha=0.9),
                                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.8))
                            sell_count += 1
                        except (KeyError, IndexError):
                            continue
                
                signal_info = f"Signals: {buy_count} Long, {sell_count} Short"
            else:
                signal_info = "No signals available"
            
            return ax2, signal_info
        
        # Main plotting loop
        for idx, ticker in enumerate(included_tickers):
            ax = axes_flat[idx]
            try:
                # Get full price data (training + test) for this ticker
                full_price_data = quotes[ticker].dropna()
                
                # Get training and test data separately for background coloring
                training_data = training_set[ticker].dropna() if ticker in training_set.columns else pd.Series(dtype=float)
                test_data = test_set[ticker].dropna() if ticker in test_set.columns else pd.Series(dtype=float)
                
                indicators = technical_indicators[ticker]
                signals = trading_signals[ticker]
                
                # Get strategy info from best_strategies DataFrame
                if hasattr(best_strategies, 'loc') and ticker in best_strategies.index:
                    # DataFrame case
                    strategy_info = best_strategies.loc[ticker].to_dict()
                elif isinstance(best_strategies, pd.DataFrame) and 'Ticker' in best_strategies.columns:
                    # DataFrame with Ticker column
                    ticker_row = best_strategies[best_strategies['Ticker'] == ticker]
                    if not ticker_row.empty:
                        strategy_info = ticker_row.iloc[0].to_dict()
                    else:
                        strategy_info = {}
                else:
                    # Dictionary case (fallback)
                    strategy_info = best_strategies.get(ticker, {})
                
                # Add background colors for training vs test periods
                if not training_data.empty:
                    # Light blue background for training period
                    ax.axvspan(training_data.index[0], training_data.index[-1], 
                              alpha=0.1, color='lightblue', label='Training Period', zorder=0)
                
                if not test_data.empty:
                    # White background for test period (default, but explicitly set for clarity)
                    ax.axvspan(test_data.index[0], test_data.index[-1], 
                              alpha=0.05, color='white', label='Test Period', zorder=0)
                
                # Plot full price line (training + test)
                ax.plot(full_price_data.index, full_price_data.values, label='Price', color='black', linewidth=2, zorder=1)
                
                # Plot indicators and signals (full period)
                ax2, signal_info = plot_signals_and_indicators(ax, ticker, indicators, signals, strategy_info)
                
                # Create title with actual values
                strategy = strategy_info.get('Strategy', 'Unknown')
                cagr = strategy_info.get('CAGR', strategy_info.get('cagr', 0))
                volatility = strategy_info.get('Volatility', strategy_info.get('volatility', 0))
                sharpe = strategy_info.get('Best_Sharpe', strategy_info.get('sharpe_ratio', 0))
                max_dd = strategy_info.get('Max_Drawdown', strategy_info.get('max_drawdown', 0))
                short_period = strategy_info.get('Short_Period', 20)
                long_period = strategy_info.get('Long_Period', 50)
                
                title_parts = [
                    f'{ticker} - {strategy}',
                    f'CAGR: {cagr:.2%} | Vol: {volatility:.2%} | Sharpe: {sharpe:.3f} | Max DD: {max_dd:.2%} ✓',
                    f'Parameters: {short_period}/{long_period}',
                    signal_info
                ]
                
                ax.set_title('\n'.join(title_parts), fontsize=10, weight='bold')
                ax.set_ylabel('Price', fontsize=10)
                ax.legend(fontsize=8, loc='upper left')
                ax.grid(True, alpha=0.3, zorder=0)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                
                # Add a vertical line to separate training and test periods
                if not training_data.empty and not test_data.empty:
                    ax.axvline(x=training_data.index[-1], color='gray', linestyle='--', 
                              alpha=0.7, linewidth=1, label='Train/Test Split', zorder=2)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {ticker}\n{str(e)}', ha='center', va='center', 
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                print(f"Error plotting {ticker}: {e}")
        
        # Hide unused subplots
        for idx in range(len(included_tickers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # Plot the strategy signals
    print(f"\n📈 Strategy Signals by Ticker:")
    print(f"📋 Included tickers: {optimization_summary['Ticker'].tolist()}")
    plot_ticker_signals_with_annotations(
        quotes, full_trading_signals, full_technical_indicators, 
        optimization_summary, selected_tickers
    )
    
    print("\n✅ Chart generation complete!")
    
    return full_technical_indicators, full_trading_signals

def plot_portfolio_performance(optimization_summary, quotes=None, selected_tickers=None):
    """Plot portfolio performance metrics and individual ticker performance"""
    if optimization_summary is None or optimization_summary.empty:
        print("Portfolio results not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance Metrics
    metrics = ['CAGR', 'Volatility', 'Best_Sharpe', 'Max_Drawdown']
    metric_values = [optimization_summary[col].mean() * (100 if col != 'Best_Sharpe' else 1) for col in metrics]
    
    bars = axes[0,0].bar(metrics, metric_values, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
    for bar, value in zip(bars, metric_values):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f'{value:.2f}{"%" if bar.get_x() < 3 else ""}', ha='center', va='bottom', fontweight='bold')
    axes[0,0].set_title('Average Portfolio Metrics', fontweight='bold')
    axes[0,0].set_ylabel('Value (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Ticker Performance Ranking
    sorted_summary = optimization_summary.sort_values('Best_Sharpe', ascending=True)
    sharpe_range = sorted_summary['Best_Sharpe'].max() - sorted_summary['Best_Sharpe'].min()
    colors = plt.cm.RdYlGn([0.3 + 0.7 * (x - sorted_summary['Best_Sharpe'].min()) / sharpe_range for x in sorted_summary['Best_Sharpe']])
    
    axes[0,1].barh(range(len(sorted_summary)), sorted_summary['Best_Sharpe'], color=colors)
    axes[0,1].set_yticks(range(len(sorted_summary)))
    axes[0,1].set_yticklabels(sorted_summary['Ticker'])
    axes[0,1].set_xlabel('Sharpe Ratio')
    axes[0,1].set_title('Ticker Performance Ranking', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Strategy Distribution
    strategy_counts = optimization_summary['Strategy'].value_counts()
    axes[1,0].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('Strategy Distribution', fontweight='bold')
    
    # Plot 4: Risk-Return Scatter
    scatter = axes[1,1].scatter(optimization_summary['Volatility'] * 100, optimization_summary['CAGR'] * 100,
                               c=optimization_summary['Best_Sharpe'], cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
    
    for i, ticker in enumerate(optimization_summary['Ticker']):
        axes[1,1].annotate(ticker, (optimization_summary['Volatility'].iloc[i] * 100, 
                                   optimization_summary['CAGR'].iloc[i] * 100),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1,1].set_xlabel('Volatility (%)')
    axes[1,1].set_ylabel('CAGR (%)')
    axes[1,1].set_title('Risk-Return Profile', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1,1], label='Sharpe Ratio')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    best_performer = optimization_summary.loc[optimization_summary['Best_Sharpe'].idxmax()]
    summary_stats = [
        f"Number of stocks: {len(optimization_summary)}",
        f"Average CAGR: {optimization_summary['CAGR'].mean()*100:.2f}%",
        f"Average Volatility: {optimization_summary['Volatility'].mean()*100:.2f}%",
        f"Average Sharpe Ratio: {optimization_summary['Best_Sharpe'].mean():.3f}",
        f"Average Max Drawdown: {optimization_summary['Max_Drawdown'].mean()*100:.2f}%",
        f"Best performer: {best_performer['Ticker']} (Sharpe: {best_performer['Best_Sharpe']:.3f})"
    ]
    print(f"\n📊 Portfolio Summary:\n" + "\n".join(f"  • {stat}" for stat in summary_stats))

def plot_parameter_comparison(optimization_summary):
    """Plot comparison of optimized parameters across tickers"""
    if optimization_summary.empty:
        print("No optimization summary data to plot")
        return
    
    strategy_plot_configs = {
        'SMA_Cross_Signal': [('Short_Period', 'skyblue'), ('Long_Period', 'lightcoral')],
        'EMA_Cross_Signal': [('Short_Period', 'skyblue'), ('Long_Period', 'lightcoral')],
        'RSI_Signal': [('RSI_Period', 'gold'), ('Oversold_Level', 'orange'), ('Overbought_Level', 'red')],
        'ADX_Trend_Signal': [('ADX_Period', 'purple'), ('ADX_Threshold', 'indigo')]
    }
    
    for strategy, group in optimization_summary.groupby('Strategy'):
        if strategy not in strategy_plot_configs:
            continue
            
        configs = strategy_plot_configs[strategy]
        n_plots = len(configs) + 2  # +2 for Sharpe and Risk-Return plots
        cols = 2
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
        fig.suptitle(f'{strategy} - Parameter Optimization Results', fontsize=16, fontweight='bold')
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Plot parameter bars
        for i, (param, color) in enumerate(configs):
            if param in group.columns:
                axes_flat[i].bar(group['Ticker'], group[param], color=color, alpha=0.7)
                axes_flat[i].set_title(f'Optimal {param.replace("_", " ")} by Ticker')
                axes_flat[i].set_ylabel(param.replace("_", " "))
                axes_flat[i].tick_params(axis='x', rotation=45)
        
        # Sharpe ratio plot
        sharpe_idx = len(configs)
        axes_flat[sharpe_idx].bar(group['Ticker'], group['Best_Sharpe'], color='lightgreen', alpha=0.7)
        axes_flat[sharpe_idx].set_title('Best Sharpe Ratio by Ticker')
        axes_flat[sharpe_idx].set_ylabel('Sharpe Ratio')
        axes_flat[sharpe_idx].tick_params(axis='x', rotation=45)
        
        # Risk-return scatter
        scatter_idx = sharpe_idx + 1
        if scatter_idx < len(axes_flat):
            scatter = axes_flat[scatter_idx].scatter(group['Volatility'], group['CAGR'], 
                                                   c=group['Best_Sharpe'], cmap='RdYlGn', s=100, alpha=0.7)
            for i, ticker in enumerate(group['Ticker']):
                axes_flat[scatter_idx].annotate(ticker, (group['Volatility'].iloc[i], group['CAGR'].iloc[i]),
                                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes_flat[scatter_idx].set_xlabel('Volatility')
            axes_flat[scatter_idx].set_ylabel('CAGR')
            axes_flat[scatter_idx].set_title('Risk-Return Profile')
            plt.colorbar(scatter, ax=axes_flat[scatter_idx], label='Sharpe Ratio')
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def plot_optimization_metrics(optimization_summary):
    """Plot distribution of optimization metrics"""
    if optimization_summary.empty:
        print("No optimization summary data to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Optimization - Performance Metrics Distribution', fontsize=16, fontweight='bold')
    
    # Define plot configurations
    metric_configs = [
        ('Best_Sharpe', 'Sharpe Ratio', 'skyblue', 1),
        ('CAGR', 'CAGR (%)', 'lightgreen', 100),
        ('Volatility', 'Volatility (%)', 'orange', 100),
        ('Max_Drawdown', 'Max Drawdown (%)', 'red', 100)
    ]
    
    # Plot histograms
    for i, (col, title, color, multiplier) in enumerate(metric_configs):
        row, col_idx = i // 3, i % 3
        data = optimization_summary[col] * multiplier
        mean_val = data.mean()
        
        axes[row, col_idx].hist(data, bins=10, color=color, alpha=0.7, edgecolor='black')
        axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', 
                                  label=f'Mean: {mean_val:.{2 if multiplier == 100 else 3}f}{"%" if multiplier == 100 else ""}')
        axes[row, col_idx].set_title(f'{title} Distribution')
        axes[row, col_idx].set_xlabel(title)
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].legend()
    
    # Strategy distribution pie chart
    strategy_counts = optimization_summary['Strategy'].value_counts()
    axes[1,1].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Strategy Type Distribution')
    
    # Risk-return scatter by strategy
    colors = ['red', 'blue', 'green', 'orange']
    for i, strategy in enumerate(optimization_summary['Strategy'].unique()):
        strategy_data = optimization_summary[optimization_summary['Strategy'] == strategy]
        axes[1,2].scatter(strategy_data['Volatility'] * 100, strategy_data['CAGR'] * 100, 
                         c=colors[i % len(colors)], label=strategy, s=60, alpha=0.7)
    
    axes[1,2].set_xlabel('Volatility (%)')
    axes[1,2].set_ylabel('CAGR (%)')
    axes[1,2].set_title('Risk-Return Profile by Strategy')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()