import pandas as pd
import numpy as np
import statsmodels.api as sm

def find_best_benchmark_and_run_regressions(benchmark_excess_returns_df, stock_excess_returns_df):
    """
    Find the best benchmark ETF using regression analysis and then run individual stock regressions against it.
    
    Parameters:
    -----------
    benchmark_excess_returns_df : pandas.DataFrame
        DataFrame with benchmark excess returns (columns are benchmark tickers)
    stock_excess_returns_df : pandas.DataFrame
        DataFrame with stock excess returns (columns are stock tickers)
        
    Returns:
    --------
    tuple
        (best_benchmark_ticker, benchmark_stats_df, stock_regression_results_df)
    """
    # Find the best benchmark
    benchmark_stats = []
    for benchmark in benchmark_excess_returns_df.columns:
        p_values_intercept = []
        p_values_slope = []
        slopes = []
        r_squareds = []
        invalid_slope = False
        
        for stock in stock_excess_returns_df.columns:
            y = stock_excess_returns_df[stock].dropna()
            x = benchmark_excess_returns_df[benchmark].reindex(y.index)
            x = sm.add_constant(x)
            model = sm.OLS(y, x, missing='drop').fit()
            
            # Extract slope (beta)
            slope = model.params[x.columns[1]]
            if isinstance(slope, pd.Series):
                slope = slope.iloc[0]  # Extract a single value if it's a Series
            
            # Skip benchmark if beta is outside reasonable range
            if slope < 0 or slope > 2:
                invalid_slope = True
                break
                
            pval_intercept = model.pvalues['const']
            pval_slope = model.pvalues[x.columns[1]]
            p_values_intercept.append(pval_intercept)
            p_values_slope.append(pval_slope)
            slopes.append(slope)
            r_squareds.append(model.rsquared)
        
        # Skip this benchmark if any slope was invalid
        if invalid_slope or len(slopes) == 0:
            continue
            
        avg_p_intercept = np.mean(p_values_intercept)
        avg_p_slope = np.mean(p_values_slope)
        avg_slope = np.mean(slopes)
        slope_std = np.std(slopes)  # Consistency of betas
        avg_r2 = np.mean(r_squareds)
        
        # Calculate beta quality (closer to 1 is better)
        beta_quality = 1 - abs(avg_slope - 1)
        
        # Enhanced scoring formula
        r2_component = avg_r2 * 0.2  # Weight for R²
        intercept_component = (1 - avg_p_intercept) * 0.2  # Weight for intercept significance
        slope_component = (1 - avg_p_slope) * 0.2  # Weight for slope significance
        beta_quality_component = beta_quality * 0.2  # Weight for beta quality
        consistency_component = (1 / (1 + slope_std)) * 0.2  # Weight for beta consistency
        
        score = r2_component + intercept_component + slope_component + beta_quality_component + consistency_component
        
        benchmark_stats.append({
            'benchmark': benchmark,
            'avg_slope': avg_slope,
            'slope_std': slope_std,
            'avg_p_value_intercept': avg_p_intercept,
            'avg_p_value_slope': avg_p_slope,
            'avg_r_squared': avg_r2,
            'beta_quality': beta_quality,
            'score': score
        })

    benchmark_stats_df = pd.DataFrame(benchmark_stats)
    
    # Return empty results if no valid benchmarks were found
    if benchmark_stats_df.empty:
        return None, benchmark_stats_df, pd.DataFrame()
    
    best_benchmark = benchmark_stats_df.sort_values(by='score', ascending=False).iloc[0]['benchmark']

    # Run regression for each stock against the single best benchmark
    results = []
    for stock in stock_excess_returns_df.columns:
        y = stock_excess_returns_df[stock].dropna()
        x_bench = benchmark_excess_returns_df[best_benchmark].reindex(y.index)
        x = sm.add_constant(x_bench)
        model = sm.OLS(y, x, missing='drop').fit()
        
        # Extract regression results
        p_value = model.pvalues[x.columns[1]]
        r_squared = model.rsquared
        intercept = model.params.iloc[0]
        slope = model.params.iloc[1]
        
        # Calculate correlation (ensuring arrays have same length)
        common_idx = y.index.intersection(x_bench.index)
        if len(common_idx) > 1:
            y_aligned = y.loc[common_idx]
            x_aligned = x_bench.loc[common_idx]
            if len(y_aligned) == len(x_aligned) and len(y_aligned) > 0:
                # Convert to numpy arrays and ensure they're 1D arrays with same shape
                y_array = np.array(y_aligned).flatten()
                x_array = np.array(x_aligned).flatten()
                
                if y_array.shape == x_array.shape:
                    correlation = np.corrcoef(y_array, x_array)[0, 1]
                else:
                    correlation = float('nan')
            else:
                correlation = float('nan')
        else:
            correlation = float('nan')
        
        results.append({
            'Equity': stock,
            'Benchmark': best_benchmark,
            'intercept (alpha)': intercept,
            'slope (beta)': slope,
            'correlation': correlation,
            'r_squared': r_squared,
            'p_value_slope': model.pvalues[x.columns[1]],
            'p_value_intercept': model.pvalues['const']
        })

    best_benchmarks_df = pd.DataFrame(results)
    return best_benchmark, benchmark_stats_df, best_benchmarks_df