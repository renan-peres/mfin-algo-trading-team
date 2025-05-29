# Team 8 - Algo Trading (Buy & Hold Strategy)

## Poject Overview
This project implements a Buy and Hold investment strategy using Python, focusing on long-term equity investments. The strategy is designed to select high-quality stocks based on fundamental analysis and optimize the portfolio for maximum returns.

```mermaid
flowchart TD
    Title[Buy and Hold - Long Term Investment Strategy]
    
    Title --> A[Data Collection Process]
    A --> B[scrape_tickers.ipynb]
    
    B --> B1[Scrape S&P 500 Constituents]
    B1 --> B2[equity_tickers.txt]
    
    B2 --> C[scrape_fundamentals.ipynb]
    B2 --> D[scrape_quotes.ipynb]
    
    %% Add invisible spacing nodes for alignment
    C ~~~ D
    
    C --> C1[Scrape Company Fundamentals]
    C1 --> C2[Screening Criteria]
    
    C2 --> C3[Market Cap > $50B]
    C2 --> C4[P/E < 30]
    C2 --> C5[P/S ≤ 5]
    C2 --> C6[P/B > 0 & P/B < 10]
    C2 --> C7[Operating Margin > 20%]
    
    C3 --> C8[Screened Fundamentals Dataset]
    C4 --> C8
    C5 --> C8
    C6 --> C8
    C7 --> C8
    
    D --> D1[Scrape Historical Price Data]
    
    C8 --> E[Data Integration]
    D1 --> E
    
    E --> F[Portfolio Construction Pipeline]
    F --> G[01_equity_portfolio_construction.ipynb]
    
    G --> G1["Markowitz Mean Variance - Sharpe Ratio Optimization"]
    G1 --> G2[Min Assets: 5]
    G1 --> G3[Max Assets: ∞]
    G1 --> G4[Max Asset per Sector: 2]
    G1 --> G5[Max Allocation: 30%]
    G1 --> G6[Min Allocation: 5%]
    
    G2 --> H["Backtest - bt.Strategy()"]
    G3 --> H
    G4 --> H
    G5 --> H
    G6 --> H
    
    H --> H1["bt.algos.RunMonthly() - Execute Monthly"]
    H --> H2["bt.algos.SelectAll() - Include All Assets"]
    H --> H3["bt.algos.WeighSpecified(**weights) - Asset Optimal Weights"]
    H --> H4["bt.algos.Rebalance() - Rebalance Portfolio"]
    
    %% Styling
    style Title fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#ffffff
    style A fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#ffffff
    style B fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style C fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style D fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#ffffff
    style C2 fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#ffffff
    style C8 fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    style E fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#ffffff
    style F fill:#27ae60,stroke:#229954,stroke-width:2px,color:#ffffff
    style G1 fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#ffffff
    style H fill:#8e44ad,stroke:#7d3c98,stroke-width:2px,color:#ffffff
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