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

STRATEGY_NAMES = ['SMA_Cross_Signal', 'EMA_Cross_Signal', 'ADX_Trend_Signal', 'RSI_Signal']

# ===============================================================================
# TECHNICAL INDICATORS & SIGNALS
# ===============================================================================

def calculate_technical_indicators(prices_df, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """Calculate SMA and EMA technical indicators with customizable timeperiods"""
    indicators = {}
    min_required = max(sma_long, ema_long) + 10
    
    for ticker in prices_df.columns:       
        close = prices_df[ticker].dropna()
        if len(close) < min_required:
            continue
            
        ticker_indicators = pd.DataFrame(index=close.index)
        ticker_indicators['Close'] = close
        
        try:
            ticker_indicators[f'SMA_{sma_short}'] = talib.SMA(close, timeperiod=sma_short)
            ticker_indicators[f'SMA_{sma_long}'] = talib.SMA(close, timeperiod=sma_long)
            ticker_indicators[f'EMA_{ema_short}'] = talib.EMA(close, timeperiod=ema_short)
            ticker_indicators[f'EMA_{ema_long}'] = talib.EMA(close, timeperiod=ema_long)
            indicators[ticker] = ticker_indicators
        except Exception:
            continue
    return indicators

def generate_trading_signals(indicators_dict, sma_short=50, sma_long=200, ema_short=26, ema_long=52):
    """Generate buy/sell signals based on SMA and EMA crossovers"""
    strategies = {}
    for ticker, indicators in indicators_dict.items():
        try:
            ticker_signals = pd.DataFrame(index=indicators.index)
            ticker_signals['SMA_Cross_Signal'] = np.where(
                indicators[f'SMA_{sma_short}'] > indicators[f'SMA_{sma_long}'], 1, 
                np.where(indicators[f'SMA_{sma_short}'] < indicators[f'SMA_{sma_long}'], -1, 0)
            )
            ticker_signals['EMA_Cross_Signal'] = np.where(
                indicators[f'EMA_{ema_short}'] > indicators[f'EMA_{ema_long}'], 1, 
                np.where(indicators[f'EMA_{ema_short}'] < indicators[f'EMA_{ema_long}'], -1, 0)
            )
            strategies[ticker] = ticker_signals
        except Exception:
            continue
    return strategies

# ===============================================================================
# BACKTESTING UTILITIES
# ===============================================================================

def safe_get_stat(stats, strategy_col, stat_names, default=0):
    """Safely extract statistics with fallback options"""
    for stat_name in stat_names:
        if stat_name in stats.index:
            value = stats.loc[stat_name, strategy_col]
            return value if not pd.isna(value) else default
    return default

def run_backtest_silent(backtest):
    """Run backtest while suppressing output"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = StringIO()
        return bt.run(backtest)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def extract_performance_metrics(result, strategy_name):
    """Extract key performance metrics from backtest result"""
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
    except Exception:
        return None

# ===============================================================================
# OPTIMIZATION ENGINE - CONSOLIDATED
# ===============================================================================

def calculate_performance_metrics(returns):
    """Calculate all performance metrics from returns array"""
    if len(returns) < 5:
        return None
        
    total_return = np.prod(1 + returns) - 1
    volatility = np.std(returns) * np.sqrt(252)
    mean_return = np.mean(returns) * 252
    
    if volatility <= 1e-10:
        return None
        
    sharpe = mean_return / volatility
    
    # Sortino ratio
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        sortino = mean_return / downside_deviation if downside_deviation > 1e-10 else sharpe
    else:
        sortino = sharpe if mean_return > 0 else 0
    
    # Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / running_max) - 1
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # CAGR
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    if not all(np.isfinite([sortino, sharpe, cagr])):
        return None
        
    return {
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'total_return': total_return
    }

def optimize_strategy_parameters(data, parameter_ranges, strategy_type):
    """Optimize parameters for a single strategy type"""
    results = []
    
    if strategy_type == 'SMA_Cross_Signal':
        combinations = product(parameter_ranges['short_periods'], parameter_ranges['long_periods'])
        for short_period, long_period in combinations:
            if short_period >= long_period:
                continue
                
            # Calculate moving averages
            ma_short = data.rolling(short_period).mean()
            ma_long = data.rolling(long_period).mean()
            
            # Create target weights
            target_weights = ma_long.copy()
            target_weights[ma_short > ma_long] = 1.0
            target_weights[ma_short <= ma_long] = -1.0
            target_weights = target_weights.fillna(0)
            
            # Calculate returns
            returns = data.pct_change().fillna(0)
            strategy_returns = (returns * target_weights.shift(1)).fillna(0)
            valid_returns = strategy_returns[~np.isnan(strategy_returns)]
            
            metrics = calculate_performance_metrics(valid_returns)
            if metrics:
                metrics.update({
                    'strategy_type': strategy_type,
                    'short_period': short_period,
                    'long_period': long_period
                })
                results.append(metrics)
                
    elif strategy_type == 'EMA_Cross_Signal':
        combinations = product(parameter_ranges['short_periods'], parameter_ranges['long_periods'])
        for short_period, long_period in combinations:
            if short_period >= long_period:
                continue
                
            # Calculate exponential moving averages
            ma_short = data.ewm(span=short_period).mean()
            ma_long = data.ewm(span=long_period).mean()
            
            # Create target weights
            target_weights = ma_long.copy()
            target_weights[ma_short > ma_long] = 1.0
            target_weights[ma_short <= ma_long] = -1.0
            target_weights = target_weights.fillna(0)
            
            # Calculate returns
            returns = data.pct_change().fillna(0)
            strategy_returns = (returns * target_weights.shift(1)).fillna(0)
            valid_returns = strategy_returns[~np.isnan(strategy_returns)]
            
            metrics = calculate_performance_metrics(valid_returns)
            if metrics:
                metrics.update({
                    'strategy_type': strategy_type,
                    'short_period': short_period,
                    'long_period': long_period
                })
                results.append(metrics)
    
    return results

def optimize_single_ticker_parameters(args):
    """Optimized single ticker parameter optimization"""
    ticker, ticker_data, parameter_ranges = args
    
    if len(ticker_data) < 250:
        return ticker, None
    
    data = ticker_data[ticker].dropna()
    all_results = []
    
    # Optimize both strategies
    for strategy in ['SMA_Cross_Signal', 'EMA_Cross_Signal']:
        strategy_results = optimize_strategy_parameters(data, parameter_ranges[strategy], strategy)
        all_results.extend(strategy_results)
    
    if not all_results:
        return ticker, None
    
    results_df = pd.DataFrame(all_results)
    best_idx = results_df['sortino_ratio'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    return ticker, {
        'best_strategy': best_params['strategy_type'],
        'best_params': best_params,
        'strategy_type': best_params['strategy_type'],
        'results_df': results_df
    }

def find_optimal_portfolio_with_parameter_optimization(quotes, min_cagr=0.0, max_volatility=0.3, max_stocks=10, n_jobs=None, heatmap_metric='sortino', parameter_ranges=None):
    """Optimized portfolio optimization with consolidated logic"""
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, len(quotes.columns))
    
    if parameter_ranges is None:
        parameter_ranges = {
            'SMA_Cross_Signal': {
                'short_periods': list(range(10, 45, 5)),
                'long_periods': list(range(60, 200, 20))
            },
            'EMA_Cross_Signal': {
                'short_periods': list(range(10, 30, 5)),
                'long_periods': list(range(35, 65, 5))
            }
        }
    
    # Prepare data for parallel processing
    ticker_data_list = []
    for ticker in quotes.columns:
        ticker_data = quotes[[ticker]].dropna()
        if len(ticker_data) >= 250:
            ticker_data_list.append((ticker, ticker_data, parameter_ranges))
    
    # Parallel optimization
    all_optimization_results = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_ticker = {
            executor.submit(optimize_single_ticker_parameters, args): args[0] 
            for args in ticker_data_list
        }
        for future in as_completed(future_to_ticker):
            ticker, result = future.result()
            if result is not None:
                all_optimization_results[ticker] = result
    
    if not all_optimization_results:
        return None
    
    # Filter and select top performers
    filter_data = [{
        'ticker': ticker,
        'cagr': results['best_params']['cagr'],
        'volatility': results['best_params']['volatility'],
        'sharpe_ratio': results['best_params']['sharpe_ratio'],
        'sortino_ratio': results['best_params']['sortino_ratio']
    } for ticker, results in all_optimization_results.items()]
    
    filter_df = pd.DataFrame(filter_data)
    qualified_mask = (filter_df['cagr'] >= min_cagr) & (filter_df['volatility'] <= max_volatility)
    qualified_df = filter_df[qualified_mask]
    
    if qualified_df.empty:
        return None
    
    top_df = qualified_df.nlargest(max_stocks, 'sortino_ratio')
    top_tickers = top_df['ticker'].tolist()
    
    # Generate portfolio weights and signals
    portfolio_weights = pd.DataFrame(index=quotes.index, columns=quotes.columns).fillna(0)
    optimized_signals = {}
    magnitude = 1.0 / len(top_tickers)
    
    for ticker in top_tickers:
        params = all_optimization_results[ticker]['best_params']
        ticker_data = quotes[ticker].dropna()
        
        # Calculate moving averages based on strategy type
        if params['strategy_type'] == 'SMA_Cross_Signal':
            ma_short = ticker_data.rolling(params['short_period']).mean()
            ma_long = ticker_data.rolling(params['long_period']).mean()
        else:  # EMA_Cross_Signal
            ma_short = ticker_data.ewm(span=params['short_period']).mean()
            ma_long = ticker_data.ewm(span=params['long_period']).mean()
        
        # Create target weights
        target_weights = ma_long.copy()
        target_weights[ma_short > ma_long] = magnitude
        target_weights[ma_short <= ma_long] = -magnitude
        target_weights = target_weights.fillna(0)
        
        # Store signals and weights
        ticker_signals = pd.DataFrame(index=ticker_data.index)
        ticker_signals[params['strategy_type']] = np.where(target_weights > 0, 1, 
                                                         np.where(target_weights < 0, -1, 0))
        optimized_signals[ticker] = ticker_signals
        
        reindexed_weights = target_weights.reindex(quotes.index, method='ffill').fillna(0)
        portfolio_weights[ticker] = reindexed_weights
    
    # Calculate portfolio performance
    portfolio_returns = []
    for i in range(1, len(quotes)):
        daily_return = 0
        for ticker in top_tickers:
            if ticker in quotes.columns:
                try:
                    current_price = quotes[ticker].iloc[i]
                    previous_price = quotes[ticker].iloc[i-1]
                    price_return = (current_price / previous_price) - 1
                    weight = portfolio_weights[ticker].iloc[i-1]
                    daily_return += price_return * weight
                except (IndexError, KeyError, ZeroDivisionError):
                    continue
        portfolio_returns.append(daily_return)
    
    portfolio_returns = np.array(portfolio_returns)
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    
    # Calculate portfolio statistics
    if len(portfolio_returns) > 0:
        portfolio_total_return = np.prod(1 + portfolio_returns) - 1
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_sharpe = (np.mean(portfolio_returns) * 252) / (portfolio_volatility + 1e-10)
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)
        portfolio_running_max = np.maximum.accumulate(portfolio_cumulative)
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1
        portfolio_max_drawdown = np.min(portfolio_drawdown)
        years = len(portfolio_returns) / 252
        portfolio_cagr = (1 + portfolio_total_return) ** (1/years) - 1 if years > 0 else 0
    else:
        portfolio_cagr = portfolio_sharpe = portfolio_volatility = portfolio_max_drawdown = 0
        portfolio_total_return = 0
    
    portfolio_stats = {
        'cagr': portfolio_cagr,
        'total_return': portfolio_total_return,
        'sharpe_ratio': portfolio_sharpe,
        'volatility': portfolio_volatility,
        'max_drawdown': portfolio_max_drawdown
    }
    
    # Create summary results
    optimization_summary = [{
        'Ticker': ticker,
        'Strategy': all_optimization_results[ticker]['best_params']['strategy_type'],
        'Best_Sharpe': all_optimization_results[ticker]['best_params']['sharpe_ratio'],
        'Best_Sortino': all_optimization_results[ticker]['best_params']['sortino_ratio'],
        'CAGR': all_optimization_results[ticker]['best_params']['cagr'],
        'Max_Drawdown': all_optimization_results[ticker]['best_params']['max_drawdown'],
        'Volatility': all_optimization_results[ticker]['best_params']['volatility'],
        'Short_Period': all_optimization_results[ticker]['best_params']['short_period'],
        'Long_Period': all_optimization_results[ticker]['best_params']['long_period']
    } for ticker in top_tickers]
    
    best_strategies_dict = {ticker: {
        'strategy': all_optimization_results[ticker]['best_params']['strategy_type'],
        'cagr': all_optimization_results[ticker]['best_params']['cagr'],
        'total_return': all_optimization_results[ticker]['best_params'].get('total_return', 0),
        'max_drawdown': all_optimization_results[ticker]['best_params']['max_drawdown'],
        'sharpe_ratio': all_optimization_results[ticker]['best_params']['sharpe_ratio'],
        'sortino_ratio': all_optimization_results[ticker]['best_params']['sortino_ratio'],
        'volatility': all_optimization_results[ticker]['best_params']['volatility']
    } for ticker in top_tickers}
    
    return {
        'portfolio_stats': portfolio_stats,
        'portfolio_weights': portfolio_weights,
        'optimization_summary': pd.DataFrame(optimization_summary),
        'best_strategies_df': pd.DataFrame.from_dict(best_strategies_dict, orient='index'),
        'selected_tickers': top_tickers,
        'optimized_signals': optimized_signals,
        'all_optimization_results': all_optimization_results,
        'processing_time': 'Fast parallel processing completed',
        'heatmap_metric': heatmap_metric,
        'parameter_ranges': parameter_ranges
    }

# ===============================================================================
# PLOTTING FUNCTIONS
# ===============================================================================

def plot_optimization_heatmaps(optimization_results, metric=None):
    """Simplified heatmap plotting with enhanced titles"""
    import seaborn as sns
    
    if metric is None:
        metric = optimization_results.get('heatmap_metric', 'sortino')
    
    metric_mapping = {'cagr': 'cagr', 'sharpe': 'sharpe_ratio', 'sortino': 'sortino_ratio'}
    metric_column = metric_mapping.get(metric, 'sortino_ratio')
    metric_display = metric.title()
    
    for ticker in optimization_results['selected_tickers']:
        if ticker not in optimization_results['all_optimization_results']:
            continue
        
        results = optimization_results['all_optimization_results'][ticker]
        results_df = results['results_df']
        strategy_type = results['strategy_type']
        best_params = results['best_params']
        
        strategy_filtered_df = results_df[results_df['strategy_type'] == strategy_type].copy()
        valid_results_df = strategy_filtered_df[
            (~strategy_filtered_df[metric_column].isna()) & 
            (~strategy_filtered_df[metric_column].isin([np.inf, -np.inf]))
        ].copy()
        
        if valid_results_df.empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        pivot_table = valid_results_df.pivot_table(
            values=metric_column, 
            index='long_period', 
            columns='short_period',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', 
                   square=False, linewidths=0.5)
        
        # Invert y-axis
        plt.gca().invert_yaxis()
        
        # Enhanced title with best parameters and metrics
        best_cagr = best_params['cagr']
        best_sharpe = best_params['sharpe_ratio']
        best_sortino = best_params['sortino_ratio']
        
        title_line1 = f'{ticker} - {strategy_type} Parameter Optimization ({metric_display}-Based)'
        title_line2 = f'Best Parameters: {best_params["short_period"]}/{best_params["long_period"]}'
        title_line3 = f'(CAGR: {best_cagr:.1%}, Sharpe: {best_sharpe:.3f}, Sortino: {best_sortino:.3f})'
        
        plt.title(f'{title_line1}\n{title_line2}\n{title_line3}',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel('Short Period')
        plt.ylabel('Long Period')
        
        # Highlight best combination
        best_short = best_params['short_period']
        best_long = best_params['long_period']
        if best_short in pivot_table.columns and best_long in pivot_table.index:
            short_idx = list(pivot_table.columns).index(best_short)
            long_idx = list(pivot_table.index).index(best_long)
            rect = plt.Rectangle((short_idx, long_idx), 1, 1, 
                               fill=False, edgecolor='red', linewidth=3)
            plt.gca().add_patch(rect)
        
        plt.tight_layout()
        plt.show()
        
def plot_complete_strategy_analysis(optimization_results, quotes, training_set, test_set, cols=2, rows=None):
    """Complete function to generate and plot optimized trading signals for full dataset"""
    selected_tickers = optimization_results['selected_tickers']
    optimization_summary = optimization_results['optimization_summary']
    
    # Generate signals for full dataset
    full_technical_indicators = {}
    full_trading_signals = {}
    
    for ticker in selected_tickers:
        if ticker in optimization_results['all_optimization_results']:
            params = optimization_results['all_optimization_results'][ticker]['best_params']
            strategy_type = params['strategy_type']
            ticker_full_data = quotes[[ticker]].dropna()
            
            ticker_indicators = pd.DataFrame(index=ticker_full_data.index)
            ticker_indicators['Close'] = ticker_full_data[ticker]
            
            if strategy_type == 'SMA_Cross_Signal':
                short_period = params['short_period']
                long_period = params['long_period']
                ticker_indicators[f'SMA_{short_period}'] = ticker_full_data[ticker].rolling(short_period).mean()
                ticker_indicators[f'SMA_{long_period}'] = ticker_full_data[ticker].rolling(long_period).mean()
                
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                position = np.where(
                    ticker_indicators[f'SMA_{short_period}'] > ticker_indicators[f'SMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'SMA_{short_period}'] < ticker_indicators[f'SMA_{long_period}'], -1, 0)
                )
                ticker_signals['SMA_Cross_Signal'] = position
                
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)
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
                
                ticker_signals = pd.DataFrame(index=ticker_full_data.index)
                position = np.where(
                    ticker_indicators[f'EMA_{short_period}'] > ticker_indicators[f'EMA_{long_period}'], 1,
                    np.where(ticker_indicators[f'EMA_{short_period}'] < ticker_indicators[f'EMA_{long_period}'], -1, 0)
                )
                ticker_signals['EMA_Cross_Signal'] = position
                
                position_series = pd.Series(position, index=ticker_full_data.index)
                signal_changes = position_series.diff()
                ticker_signals['Buy_Signal'] = np.where(signal_changes == 2, 1, 0)
                ticker_signals['Sell_Signal'] = np.where(signal_changes == -2, 1, 0)
                ticker_signals['Buy_Signal'] = np.where(
                    (signal_changes == 1) & (position_series == 1), 1, ticker_signals['Buy_Signal']
                )
                ticker_signals['Sell_Signal'] = np.where(
                    (signal_changes == -1) & (position_series == -1), 1, ticker_signals['Sell_Signal']
                )
            
            full_technical_indicators[ticker] = ticker_indicators
            full_trading_signals[ticker] = ticker_signals
    
    # Plot the results with minimal plotting function
    if not selected_tickers:
        return full_technical_indicators, full_trading_signals
    
    cols = min(cols, len(selected_tickers))
    if rows is None:
        rows = (len(selected_tickers) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    axes_flat = [axes] if len(selected_tickers) == 1 else axes.flatten()
    
    for idx, ticker in enumerate(selected_tickers):
        ax = axes_flat[idx]
        try:
            full_price_data = quotes[ticker].dropna()
            training_data = training_set[ticker].dropna() if ticker in training_set.columns else pd.Series(dtype=float)
            test_data = test_set[ticker].dropna() if ticker in test_set.columns else pd.Series(dtype=float)
            
            indicators = full_technical_indicators[ticker]
            signals = full_trading_signals[ticker]
            
            strategy_info = optimization_summary[optimization_summary['Ticker'] == ticker].iloc[0].to_dict() if not optimization_summary[optimization_summary['Ticker'] == ticker].empty else {}
            
            # Add background colors
            if not training_data.empty:
                ax.axvspan(training_data.index[0], training_data.index[-1], 
                        alpha=0.1, color='#93cbf9', label='Training Period', zorder=0)
            if not test_data.empty:
                ax.axvspan(test_data.index[0], test_data.index[-1], 
                        alpha=0.05, color='white', label='Test Period', zorder=0)
            
            # Plot price
            ax.plot(full_price_data.index, full_price_data.values, label='Price', color='black', linewidth=2, zorder=1)
            
            # Plot indicators and signals
            strategy = strategy_info.get('Strategy', 'SMA_Cross_Signal')
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
            
            # Plot signals
            buy_signals = signals.get('Buy_Signal', pd.Series(dtype=float, index=signals.index))
            sell_signals = signals.get('Sell_Signal', pd.Series(dtype=float, index=signals.index))
            
            buy_count = sell_count = 0
            buy_dates = buy_signals[buy_signals == 1].index
            for date in buy_dates:
                if date in indicators.index:
                    try:
                        y_val = indicators.loc[date, f'SMA_{short_period}'] if strategy == 'SMA_Cross_Signal' else indicators.loc[date, f'EMA_{short_period}']
                        ax.scatter(date, y_val, color='green', marker='^', s=120, alpha=0.9, zorder=6)
                        buy_count += 1
                    except (KeyError, IndexError):
                        continue
            
            sell_dates = sell_signals[sell_signals == 1].index
            for date in sell_dates:
                if date in indicators.index:
                    try:
                        y_val = indicators.loc[date, f'SMA_{short_period}'] if strategy == 'SMA_Cross_Signal' else indicators.loc[date, f'EMA_{short_period}']
                        ax.scatter(date, y_val, color='red', marker='v', s=120, alpha=0.9, zorder=6)
                        sell_count += 1
                    except (KeyError, IndexError):
                        continue
            
            signal_info = f"Signals: {buy_count} Long, {sell_count} Short"
            title_parts = [f'{ticker} - {strategy}', signal_info]
            ax.set_title('\n'.join(title_parts), fontsize=10, weight='bold')
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(True, alpha=0.3, zorder=0)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            
            if not training_data.empty and not test_data.empty:
                ax.axvline(x=training_data.index[-1], color='red', linestyle='--', 
                        alpha=0.7, linewidth=1.5, label='Train/Test Split', zorder=2)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting {ticker}\n{str(e)}', ha='center', va='center', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(selected_tickers), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return full_technical_indicators, full_trading_signals

def plot_portfolio_performance(optimization_summary, cols=2, rows=1):
    """Plot portfolio performance analysis with descriptive statistics table and risk-return profile"""
    stats_data = []
    metrics = ['CAGR', 'Volatility', 'Best_Sharpe', 'Best_Sortino', 'Max_Drawdown']
    
    for metric in metrics:
        if metric in optimization_summary.columns:
            col_data = optimization_summary[metric]
            display_name = metric.replace('_', ' ')
            if metric == 'Best_Sharpe':
                display_name = 'Sharpe Ratio'
            elif metric == 'Best_Sortino':
                display_name = 'Sortino Ratio'
            
            stats_data.append({
                'Metric': display_name,
                'Mean': col_data.mean(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Count': col_data.count()
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Descriptive Statistics Table
    ax1 = axes[0]
    ax1.axis('tight')
    ax1.axis('off')
    
    asset_count = len(optimization_summary)
    ax1.set_title(f'Portfolio Performance Statistics\nAssets: {asset_count}', 
                  fontsize=14, weight='bold', pad=20)

    table_data = []
    for _, row in stats_df.iterrows():
        metric = row['Metric']
        if metric in ['CAGR', 'Volatility', 'Max Drawdown']:
            formatted_row = [metric, f"{row['Mean']:.2%}", f"{row['Min']:.2%}", f"{row['Max']:.2%}"]
        else:
            formatted_row = [metric, f"{row['Mean']:.3f}", f"{row['Min']:.3f}", f"{row['Max']:.3f}"]
        table_data.append(formatted_row)
    
    table = ax1.table(
        cellText=table_data,
        colLabels=['Metric', 'Mean', 'Min', 'Max'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    num_table_columns = 4
    for i in range(num_table_columns):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i, (_, row) in enumerate(stats_df.iterrows(), 1):
        metric = row['Metric']
        if metric in ['Sharpe Ratio', 'Sortino Ratio']:
            color = '#E8F5E8'
        elif metric == 'CAGR':
            color = '#FFF9C4'
        elif metric in ['Volatility', 'Max Drawdown']:
            color = '#FFEBEE'
        else:
            color = 'white'
        
        for j in range(num_table_columns):
            table[(i, j)].set_facecolor(color)
    
    # Risk-Return Profile
    if len(axes) > 1:
        ax2 = axes[1]
        if all(col in optimization_summary.columns for col in ['Volatility', 'CAGR', 'Best_Sortino']):
            scatter = ax2.scatter(optimization_summary['Volatility'] * 100, 
                                optimization_summary['CAGR'] * 100,
                                c=optimization_summary['Best_Sortino'], 
                                cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
            
            ax2.set_title('Risk-Return Profile\n(Color = Sortino Ratio)', fontsize=14, weight='bold')
            ax2.set_xlabel('Volatility (%)')
            ax2.set_ylabel('CAGR (%)')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Sortino Ratio', rotation=270, labelpad=20)
            
            if 'Ticker' in optimization_summary.columns:
                for i, ticker in enumerate(optimization_summary['Ticker']):
                    ax2.annotate(ticker, 
                               (optimization_summary['Volatility'].iloc[i] * 100, 
                                optimization_summary['CAGR'].iloc[i] * 100),
                               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Risk-Return Profile\nRequires: Volatility, CAGR, Best_Sortino columns',
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    for idx in range(2, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return stats_df