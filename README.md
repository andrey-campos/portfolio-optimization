# Portfolio Optimization Tool

A robust Python library for modern portfolio optimization and analysis with advanced algorithm implementations and computer vision integration.

## Features

- **Multiple Optimization Strategies**
 - Sharpe Ratio Maximization
 - Robust Optimization (accounts for uncertainty in return estimates)
 - Quantum Algorithm for Portfolio Optimization (QAOA simulation)

- **Computer Vision Integration (soon)** 
 - Extract ticker symbols from financial documents and sector reports
 - Process financial charts to identify sector allocations
 - Convert visual data into structured portfolio inputs

- **Comprehensive Analysis**
 - Statistical analysis of portfolio performance metrics
 - Correlation heatmaps and asset allocation visualization
 - Volatility and expected return calculations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt
```

## Multiple Methods of Quick Start 
```bash 
from portfolio_optimizer import Portfolio

# Initialize portfolio with default settings
portfolio = Portfolio(num_stocks=5, start_date="2022-08-08", end_date="2023-08-12")

# Process portfolio data and run optimization
results = portfolio.process_portfolio()

# Access optimization results
print(f"Portfolio Expected Return: {results.expected_returns}")
print(f"Portfolio Volatility: {results.volatility}")

# alternatively get individual results
portfolio.expected_returns(want_expected_returns=True)

# Visualize results
portfolio.asset_piechart(results.portfolio)
portfolio.correlation_heatmap(results.portfolio)
# OR

# uses itself even without explicitly passing it in as a argument.
portfolio.asset_piechart()
portfolio.correlation_heatmap()

## Advanced Customization 
``` bash 
# Create a portfolio with custom parameters
portfolio = Portfolio(
    num_stocks=10,
    start_date="2022-01-01",
    end_date="2023-01-01",
    trading_days=252,
    annual_rate=0.04,
    entropy_lambda=0.3,
    want_diversification=True
)
```

## Using Different Optimization Strategies
``` bash
# Using Sharpe optimization 
from portfolio_optimizer.strategies import SharpeOptimization

portfolio = Portfolio(num_stocks=3, start_date="2022-08-08", end_date="2023-08-12")

sharpe_optimizer = SharpeOptimization(portfolio)
sharpe_results = sharpe_optimizer.get_optimized_portfolio()

# print entire portfolio data structure 
print(sharpe_optimizer)

# print new variance through dot notation 
print(sharpe_results.variance)

# Using robust optimization
# implementation soon..
```

## Contributing 
Contributions are all welcome! Please feel free to submit a Pull
Request or open an issue!

