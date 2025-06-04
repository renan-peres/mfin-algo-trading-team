import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

def analyze_benchmark_regression(benchmark_excess_returns_df, stock_excess_returns_df, best_benchmark, portfolio_df):
    """Perform comprehensive benchmark regression analysis with visualization."""
    
    print(f"📈 BENCHMARK REGRESSION ANALYSIS: {best_benchmark}")
    print("="*70)
    
    benchmark_returns = benchmark_excess_returns_df[best_benchmark].dropna()
    regression_results = {}
    
    for ticker in stock_excess_returns_df.columns:
        stock_returns = stock_excess_returns_df[ticker].dropna()
        aligned_data = pd.concat([benchmark_returns, stock_returns], axis=1, join='inner').dropna()
        
        if len(aligned_data) < 5:
            continue
        
        X, y = aligned_data.iloc[:, 0].values.reshape(-1, 1), aligned_data.iloc[:, 1].values
        reg_model = LinearRegression().fit(X, y)
        y_pred = reg_model.predict(X)
        
        # Calculate all metrics
        r_squared = r2_score(y, y_pred)
        correlation, p_value_corr = stats.pearsonr(X.flatten(), y)
        beta, alpha = reg_model.coef_[0], reg_model.intercept_
        n = len(X)
        f_statistic = (r_squared / 1) / ((1 - r_squared) / (n - 2))
        p_value_regression = 1 - stats.f.cdf(f_statistic, 1, n - 2)
        
        # Get portfolio weight
        portfolio_weight = portfolio_df[portfolio_df['Ticker'] == ticker]['Weight'].iloc[0] if ticker in portfolio_df['Ticker'].values else 0
        
        regression_results[ticker] = {
            'benchmark_returns': aligned_data.iloc[:, 0], 'stock_returns': aligned_data.iloc[:, 1],
            'beta': beta, 'alpha': alpha, 'r_squared': r_squared, 'correlation': correlation,
            'p_value_regression': p_value_regression, 'portfolio_weight': portfolio_weight, 'n_observations': n
        }
        
        print(f"🔍 {ticker}: β={beta:.4f}, R²={r_squared:.4f}, p={p_value_regression:.4f}, Weight={portfolio_weight:.1%}")
    
    return regression_results

def plot_benchmark_analysis(regression_results, best_benchmark):
    """Create scatter plots and comprehensive dashboard."""
    
    n_stocks = len(regression_results)
    cols = 3 if n_stocks > 6 else 2 if n_stocks > 2 else 1
    rows = (n_stocks + cols - 1) // cols
    
    # Scatter plots
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_stocks == 1: axes = [axes]
    elif rows == 1 and cols > 1: axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    elif rows > 1: axes = axes.flatten()
    else: axes = [axes]
    
    for i, (ticker, results) in enumerate(regression_results.items()):
        if i >= len(axes): continue
        ax = axes[i]
        
        x, y = results['benchmark_returns'].values, results['stock_returns'].values
        beta, r_squared, p_value, weight = results['beta'], results['r_squared'], results['p_value_regression'], results['portfolio_weight']
        
        ax.scatter(x, y, alpha=0.6, s=40, color='blue', edgecolors='black', linewidth=0.5)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, beta * x_line + results['alpha'], color='red', linewidth=2, label=f'β = {beta:.3f}')
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        ax.set_title(f'{ticker} vs {best_benchmark}\nR² = {r_squared:.4f}, p = {p_value:.4f}{significance}', fontsize=11, pad=10)
        ax.set_xlabel(f'{best_benchmark} Excess Returns')
        ax.set_ylabel(f'{ticker} Excess Returns')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        ax.text(0.05, 0.95, f'Weight: {weight:.1%}', transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    for i in range(len(regression_results), len(axes)): axes[i].set_visible(False)
    plt.tight_layout()
    plt.suptitle(f'Portfolio Assets vs {best_benchmark} Benchmark - Regression Analysis', fontsize=16, y=1.02)
    plt.show()
    
    # Summary table and dashboard
    summary_data = []
    for ticker, results in regression_results.items():
        p_val = results['p_value_regression']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        summary_data.append({
            'Ticker': ticker, 'Beta': results['beta'], 'Alpha': results['alpha'], 'R_Squared': results['r_squared'],
            'Correlation': results['correlation'], 'P_Value': results['p_value_regression'], 
            'Portfolio_Weight': results['portfolio_weight'], 'N_Observations': results['n_observations'], 'Significance': significance
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\n📊 BENCHMARK REGRESSION SUMMARY: {best_benchmark}")
    display(summary_df.round(4))
    
    # Portfolio-level statistics
    weighted_beta = (summary_df['Beta'] * summary_df['Portfolio_Weight']).sum()
    weighted_r_squared = (summary_df['R_Squared'] * summary_df['Portfolio_Weight']).sum()
    print(f"\n📈 PORTFOLIO STATISTICS: Weighted β={weighted_beta:.4f}, Weighted R²={weighted_r_squared:.4f}")
    print(f"  • Significant relationships (p<0.05): {(summary_df['P_Value'] < 0.05).sum()}/{len(summary_df)}")
    
    # Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    tickers, betas, weights, r_squared_values = summary_df['Ticker'].tolist(), summary_df['Beta'].tolist(), summary_df['Portfolio_Weight'].tolist(), summary_df['R_Squared'].tolist()
    
    # Beta chart
    colors = ['green' if 0.8 <= b <= 1.2 else 'orange' if 0.5 <= b <= 1.5 else 'red' for b in betas]
    bars1 = axes[0, 0].bar(tickers, betas, color=colors, alpha=0.7)
    axes[0, 0].set_title(f'Beta vs {best_benchmark}')
    axes[0, 0].set_ylabel('Beta')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Market Beta = 1.0')
    axes[0, 0].legend()
    for bar, weight in zip(bars1, weights):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{weight:.1%}', ha='center', va='bottom', fontsize=8)
    
    # R-squared chart
    colors_r2 = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in r_squared_values]
    axes[0, 1].bar(tickers, r_squared_values, color=colors_r2, alpha=0.7)
    axes[0, 1].set_title(f'R-squared vs {best_benchmark}')
    axes[0, 1].set_ylabel('R-squared')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beta vs Weight scatter
    scatter = axes[1, 0].scatter(weights, betas, s=100, alpha=0.7, c=r_squared_values, cmap='RdYlGn', edgecolors='black')
    axes[1, 0].set_xlabel('Portfolio Weight')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].set_title('Beta vs Portfolio Weight')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0]).set_label('R-squared')
    for i, ticker in enumerate(tickers):
        axes[1, 0].annotate(ticker, (weights[i], betas[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Significance pie chart
    significance_counts = summary_df['Significance'].value_counts()
    sig_counts = [significance_counts.get(level, 0) for level in ['', '*', '**', '***']]
    axes[1, 1].pie(sig_counts, labels=['Not Sig.', 'Sig. (*)', 'Highly Sig. (**)', 'Very Highly Sig. (***)'], 
                   autopct='%1.0f%%', startangle=90, colors=['lightcoral', 'yellow', 'orange', 'lightgreen'])
    axes[1, 1].set_title('Statistical Significance Distribution')
    
    plt.tight_layout()
    plt.suptitle(f'Benchmark Analysis Dashboard: {best_benchmark}', fontsize=16, y=1.02)
    plt.show()
    
    return summary_df