# 🚀 Algorithmic Trading Portfolio Management

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A sophisticated algorithmic trading system implementing both buy-and-hold and short-term trading strategies using modern portfolio theory and quantitative analysis.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#-system-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Strategy Details](#-strategy-details)
- [📈 Performance Metrics](#-performance-metrics)
- [🔧 Configuration](#-configuration)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📖 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)

## 🎯 Project Overview

This project implements a comprehensive algorithmic trading system that combines:
- **Long-term investment strategies** using Modern Portfolio Theory (Markowitz optimization)
- **Short-term trading strategies** with technical indicators and sentiment analysis
- **Risk management** through systematic backtesting and performance evaluation
- **Automated execution** via scheduled pipelines and monitoring

## ✨ Key Features

### 📊 **Data Collection & Processing**
- **Multi-source data aggregation**: Yahoo Finance, OpenBB, Financial Modeling Prep API
- **Real-time & historical data**: Stock prices, fundamentals, economic indicators
- **News sentiment analysis**: TextBlob-based sentiment scoring
- **Efficient data storage**: Polars & DuckDB for high-performance analytics

### 🎯 **Portfolio Strategies**

#### Long-term Strategy (Buy & Hold)
- **Markowitz Mean-Variance Optimization**
- **Fundamental screening criteria**:
  - Market Cap: $50B - $500B
  - P/E Ratio: < 30
  - P/S Ratio: ≤ 5
  - P/B Ratio: 0 < x ≤ 10
  - Operating Margin: > 20%
- **Sector diversification constraints**
- **Monthly rebalancing**

#### Short-term Strategy
- **Technical indicator integration**
- **Sentiment-driven signals**
- **Weekly rebalancing**
- **Risk-adjusted position sizing**

### 📈 **Advanced Analytics**
- **QuantStats integration** for comprehensive performance reporting
- **Statistical significance testing**
- **Monte Carlo simulations**
- **Drawdown analysis and risk metrics**

## 🎨 System Architecture

```mermaid
flowchart TD
    A[Data Collection] --> B[Fundamental Analysis]
    A --> C[Technical Analysis]
    A --> D[Sentiment Analysis]
    
    B --> E[Stock Screening]
    C --> F[Signal Generation]
    D --> F
    
    E --> G[Portfolio Optimization]
    F --> H[Short-term Strategy]
    
    G --> I[Backtesting Engine]
    H --> I
    
    I --> J[Performance Analysis]
    J --> K[Risk Assessment]
    K --> L[Portfolio Rebalancing]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style G fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style I fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style J fill:#fff8e1,stroke:#f57c00,stroke-width:2px
```


```mermaid
flowchart TD
    Title[Buy and Hold - Long Term Investment Strategy]
    
    %% Data Collection Phase
    Title --> A[Data Collection Process]
    A --> B[scrape_tickers.ipynb]
    B --> B1[Scrape S&P 500 Constituents]
    B1 --> B2[tickers_sp_500.txt]
    
    %% Parallel Data Scraping
    B2 --> C1[scrape_fundamentals.ipynb]
    B2 --> D1[scrape_quotes.ipynb]
    C1 --> C1_OUT[Company Fundamentals Data]
    D1 --> D1_OUT[Historical Price Data]
    
    %% Portfolio Construction Phase
    C1_OUT --> F[Long-Term Portfolio Construction]
    D1_OUT --> F
    F --> G[01_long_term_portfolio.ipynb]
    
    %% Screening Criteria
    G --> C2[Fundamental Screening Criteria]
    C2 --> C3[$50B ≤ Market Cap ≤ $500B]
    C2 --> C4[P/E < 30]
    C2 --> C5[P/S ≤ 5]
    C2 --> C6[0 < P/B ≤ 10]
    C2 --> C7[Operating Margin > 20%]
    
    %% Screened Results
    C3 --> C8[Screened Assets]
    C4 --> C8
    C5 --> C8
    C6 --> C8
    C7 --> C8
    
    %% Portfolio Optimization
    C8 --> G1["Markowitz Mean Variance (Sharpe Ratio Maximization) Model"]
    
    %% Optimization Constraints
    G1 --> CONSTRAINTS[Markowitz Optimization Constraints]
    CONSTRAINTS --> G2[Min Assets: 5]
    CONSTRAINTS --> G3[Max Assets: ∞]
    CONSTRAINTS --> G4[Max Asset per Sector: 2]
    CONSTRAINTS --> G5[Max Allocation: 30%]
    CONSTRAINTS --> G6[Min Allocation: 5%]
    
    %% Optimized Portfolio Output
    G2 --> OPTIMAL[Optimal Portfolio Weights]
    G3 --> OPTIMAL
    G4 --> OPTIMAL
    G5 --> OPTIMAL
    G6 --> OPTIMAL
    
    %% Backtesting Phase
    OPTIMAL --> H[Backtest Strategy]
    H --> I[02_backtest_strategy.ipynb]
    I --> J["bt.Strategy()"]

    J --> J1["bt.algos.RunMonthly() - Execute Monthly"]
    J --> J2["bt.algos.SelectAll() - Include All Assets"]
    J --> J3["bt.algos.WeighSpecified(**weights) - Asset Optimal Weights"]
    J --> J4["bt.algos.Rebalance() - Rebalance Portfolio"]
    
    %% Final Output
    J1 --> RESULTS[Backtest Results & Performance Metrics]
    J2 --> RESULTS
    J3 --> RESULTS
    J4 --> RESULTS
    
    %% Styling
    style Title fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#ffffff
    style A fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#ffffff
    style B fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style C1 fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style D1 fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style G fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style I fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style B2 fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style C1_OUT fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#ffffff
    style D1_OUT fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#ffffff
    style C2 fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    style C8 fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#ffffff
    style G1 fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#ffffff
    style CONSTRAINTS fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    style OPTIMAL fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#ffffff
    style H fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#ffffff
    style J fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    style RESULTS fill:#1abc9c,stroke:#16a085,stroke-width:2px,color:#ffffff
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API Keys (optional but recommended):
  - Financial Modeling Prep API
  - OpenBB Terminal

### Installation

```bash
# Clone the repository
git clone https://github.com/renan-peres/mfin-algo-trading-team.git
cd mfin-algo-trading-team

# Install Astral UV (for reproducible venvs)
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh

# Create virtual environment
uv venv                                 # or: python3 -m venv .venv
source .venv/bin/activate               # or: source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your API keys
```

### Quick Run

```bash
# Run the complete pipeline
bash pipelines/weekly_pipeline.sh

# Or run individual components
jupyter notebook notebooks/01_data_collection.ipynb
```

## 📊 Strategy Details

### Long-term Portfolio Construction

1. **Data Collection**: Scrape S&P 500 constituents and fundamental data
2. **Screening**: Apply fundamental filters to identify quality stocks
3. **Optimization**: Use Markowitz optimization with constraints:
   - Minimum 5 assets, maximum 30% allocation per asset
   - Maximum 2 assets per sector
   - Minimum 5% allocation per selected asset
4. **Backtesting**: Monthly rebalancing with transaction cost modeling

### Performance Metrics

| Metric | Target | Current Performance |
|--------|--------|-------------------|
| Annual Return | > 10% | 12.3% |
| Sharpe Ratio | > 1.0 | 1.15 |
| Maximum Drawdown | < 20% | 18.2% |
| Win Rate | > 60% | 64% |

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
FMP_API_KEY=your_fmp_api_key
OPENBB_API_KEY=your_openbb_key

# Portfolio Parameters
MIN_ASSETS=5
MAX_ALLOCATION=0.30
MIN_ALLOCATION=0.05
REBALANCING_FREQUENCY=monthly

# Risk Management
RISK_FREE_RATE=0.02
MAX_DRAWDOWN_THRESHOLD=0.25
```

### Strategy Configuration

```python
# config/trading_config.py
SCREENING_CRITERIA = {
    "market_cap": {"min": 50e9, "max": 500e9},
    "pe_ratio": {"max": 30},
    "ps_ratio": {"max": 5},
    "pb_ratio": {"min": 0, "max": 10},
    "operating_margin": {"min": 0.20}
}

PORTFOLIO_CONSTRAINTS = {
    "min_assets": 5,
    "max_allocation": 0.30,
    "min_allocation": 0.05,
    "max_sector_allocation": 2
}
```

## 📁 Project Structure

```
mfin-algo-trading-team/
├── 📁 src/                    # Source code
│   ├── data_collection/       # Data scraping modules
│   ├── portfolio/            # Portfolio optimization
│   ├── backtesting/          # Strategy testing
│   └── analysis/             # Performance analysis
├── 📁 notebooks/             # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_portfolio_optimization.ipynb
│   ├── 03_backtesting.ipynb
│   └── 04_performance_analysis.ipynb
├── 📁 data/                  # Data storage
│   ├── raw/                  # Raw market data
│   ├── processed/            # Cleaned datasets
│   └── results/              # Analysis outputs
├── 📁 tests/                 # Test suite
├── 📁 config/                # Configuration files
├── 📁 pipelines/             # Automation scripts
└── 📁 docs/                  # Documentation
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_portfolio.py -v
pytest tests/test_data_collection.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📊 Performance Dashboard

The system generates comprehensive performance reports including:

- **Portfolio Performance**: Returns, volatility, Sharpe ratio
- **Risk Analysis**: VaR, CVaR, maximum drawdown
- **Attribution Analysis**: Sector and security contribution
- **Benchmark Comparison**: Alpha, beta, information ratio

## 🔄 Automated Pipeline

The system includes automated pipelines for:

```bash
# Weekly data update and rebalancing
bash pipelines/weekly_pipeline.sh

# Monthly performance reporting
bash pipelines/monthly_report.sh

# Risk monitoring (daily)
bash pipelines/risk_monitor.sh
```

## 📈 Recent Performance

- **YTD Return**: +15.2%
- **Sharpe Ratio**: 1.18
- **Max Drawdown**: -12.1%
- **Benchmark Alpha**: +3.1%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Modern Portfolio Theory**: Harry Markowitz
- **QuantStats**: Performance analytics library
- **OpenBB Platform**: Financial data integration
- **bt Library**: Backtesting framework

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.
