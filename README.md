# Team 8 - Algo Trading (Buy & Hold Strategy)

## Poject Overview
This project implements a Buy and Hold investment strategy using Python, focusing on long-term equity investments. The strategy is designed to select high-quality stocks based on fundamental analysis and optimize the portfolio for maximum returns.

```mermaid
flowchart TD
    Title[Buy and Hold - Long Term Investment Strategy]
    
    %% Data Collection Phase
    Title --> A[Data Collection Process]
    A --> B[scrape_tickers.ipynb]
    B --> B1[Scrape S&P 500 Constituents]
    B1 --> B2[equity_tickers.txt]
    
    %% Parallel Data Scraping
    B2 --> C1[scrape_fundamentals.ipynb]
    B2 --> D1[scrape_quotes.ipynb]
    C1 --> C1_OUT[Company Fundamentals Data]
    D1 --> D1_OUT[Historical Price Data]
    
    %% Portfolio Construction Phase
    C1_OUT --> F[Portfolio Construction Pipeline]
    D1_OUT --> F
    F --> G[01_portfolio_construction.ipynb]
    
    %% Screening Criteria
    G --> C2[Fundamental Screening Criteria]
    C2 --> C3[Market Cap > $50B]
    C2 --> C4[P/E < 30]
    C2 --> C5[P/S ≤ 5]
    C2 --> C6[P/B > 0 & P/B < 10]
    C2 --> C7[Operating Margin > 20%]
    
    %% Screened Results
    C3 --> C8[Screened Assets]
    C4 --> C8
    C5 --> C8
    C6 --> C8
    C7 --> C8
    
    %% Portfolio Optimization
    C8 --> G1[Markowitz Mean Variance Optimization<br/>Sharpe Ratio Maximization]
    
    %% Optimization Constraints
    G1 --> CONSTRAINTS[Portfolio Constraints]
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

## Getting Started

```bash
# Clone the repository
git clone https://github.com/renan-peres/mfin-portfolio-management.git
cd mfin-portfolio-management

# Install Astral UV (for reproducible venvs)
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
uv venv                                 # or: python3 -m venv .venv
source .venv/bin/activate               # or: source venv/bin/activate 

# Install dependencies
uv pip install -r requirements.txt 
```