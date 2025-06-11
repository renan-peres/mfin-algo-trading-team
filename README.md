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
- [🔧 Configuration](#-configuration)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📖 Documentation](#-documentation)

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
flowchart LR
    A[Data Collection<br>(Market Data, Fundamentals, Sentiment)] 
        --> B[Feature Engineering<br>& Data Processing]
    B --> C[Strategy Layer]
    C --> D1[Long-term Portfolio<br>(Markowitz, Screening)]
    C --> D2[Short-term Signals<br>(Technical, Sentiment)]
    D1 & D2 --> E[Execution Engine<br>(Backtesting, Live Trading)]
    E --> F[Performance & Risk Analysis]
    F --> G[Reporting<br>& Rebalancing]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style D1 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style D2 fill:#e3fcec,stroke:#229954,stroke-width:2px
    style E fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style F fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    style G fill:#f9ebea,stroke:#c0392b,stroke-width:2px
```


```mermaid
flowchart TD
    A[Scrape S&P 500<br>Constituents & Data]
      --> B[Apply Fundamental<br>Screening]
      --> C[Portfolio Optimization<br>(Markowitz)]
      --> D[Backtest & Rebalance]
      --> E[Performance Evaluation]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style B fill:#e3fcec,stroke:#229954,stroke-width:2px
    style C fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style D fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    style E fill:#f9ebea,stroke:#c0392b,stroke-width:2px
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

## 🔄 Automated Pipelines

The system includes automated pipelines for:

```bash
# Weekly data update and rebalancing
bash pipelines/weekly_pipeline.sh

# Monthly performance reporting
bash pipelines/monthly_report.sh

# Risk monitoring (daily)
bash pipelines/risk_monitor.sh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Modern Portfolio Theory**: Harry Markowitz
- **QuantStats**: Performance analytics library
- **OpenBB Platform**: Financial data integration
- **bt Library**: Backtesting framework

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.
