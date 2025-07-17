import numpy as np 
import pandas as pd 
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from collections import namedtuple

class MCOptimization:
    def __init__(self, portfolio):
        """
        Initialize by getting all the Portfolio's class
        attributes through composition
        """
        self.portfolio = portfolio


    def prep_data(self) -> tuple[dict[pd.DataFrame], dict[pd.DataFrame]]:
        """
        Get daily returns by calculating percentage change then use data to
        calculate average daily returns and the covariance matrix to capture
        relationships between returns

        ### Returns:
            - mean_returns: sector based keys with each value 
            as a stock's daily avg returns (n_assets x 1)
            - cov_matrix: sector based keys with each value
            as a correlation matrix between daily avg returns (n_assets x n_assets)
            - all_returns: daily logged returns for all assets 
        """

        # get percentage change for all stocks in each sector by getting logged historical returns
        returns: dict = self.portfolio.historical_returns()
        all_returns: pd.DataFrame = pd.concat([returns[sector] for sector in returns], axis=1)
        
        mean_returns = all_returns.mean()
        cov_matrix = all_returns.cov()

        return all_returns, mean_returns, cov_matrix


    def mc_simulation(self, timeframe=10, n_sims=10, init_portfolio_value=100, st=False):
        """ 
        Perform a Monte Carlo simulation with the core assumption
        that logged daily returns follow a multivariate normal distribution 

        R_t ~ N(µ, ∑)
        - R_t: vector of daily log returns for N assets at time t
        - µ: vector of expected returns for a stock (mean returns)
        - ∑: cov matrix capturing correlations (from historical data)

        ### Args:
        - timeframe: days of simulation
        - n_sims: number of simulations to perform for optimization
        - st: returns the `fig` variable from the plot to display results for streamlit, If `st=true`

        ### Returns:
            `plt` plot that shows the possible future portfolio paths based off our
            generated correlated random returns and a initial portfolio value
        """

        _, mean_returns, cov_matrix = self.prep_data()
        mu = mean_returns.values
        n_assets = len(mu)

        # weights = (1, n) and L = (n, n) with n -> num of assets
        weights: np.array = np.random.random(n_assets)
        weights/= sum(weights)

        L = np.linalg.cholesky(cov_matrix) 
        portfolio_paths = np.zeros((timeframe, n_sims))
    
        for m in range(n_sims):
            Z = np.random.normal(size=(timeframe, n_assets)) 

            # (n, 1) + [ (t, n) * (n, n) ] = (t, n) <- random generate returns 
            correlated_returns = mu + (Z @ L.T) 

            # (t, n) * (n x 1) = (t, 1) <- returns for each day
            portfolio_returns = correlated_returns @ weights

            # calculate cumulative portfolio values V_t = V_0 × ∏(1 + r_i) 
            portfolio_paths[:, m] = init_portfolio_value * np.cumprod(1 + portfolio_returns)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(portfolio_paths)
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Days")
        ax.set_title("MC Simulation")

        if st:
            return fig 
        
        plt.show()
        return portfolio_paths

    
    def efficient_frontier(self, n_portfolios=100, st=False) -> tuple[bool, namedtuple]:
        """
        Display the efficient frontier by the generation of n random weights
        for n portfolios

        ### Args: 
        - n_portfolios: number of portfolios for efficient frontier simulation
        - st: returns the `fig` variable and `optim_results`
        data structure. Using the fig from the plot to display for streamlit, If `st=True`
        If 'st'=False it simply returns optimization results in `optim_results`
        data structure

        ### Returns 
           dictionary containing all simulated weights and simulated portfolio metrics 
           and weights array that gave the max sharpe ratio
        """

        _, mean_returns, cov_matrix = self.prep_data()

        mu = mean_returns.values
        annual_cov = (cov_matrix.values) * 252
        n_assets = len(mean_returns)

        # where we will save all of our simulation runs
        all_weights = np.zeros((n_portfolios, n_assets))
        all_returns = np.zeros(n_portfolios)
        all_volatility = np.zeros(n_portfolios)
        all_sharpe = np.zeros(n_portfolios)

        for n in range(n_portfolios):
            # generate random weights -> balance them to sum to 1 -> save them
            weights = np.array(np.random.random(n_assets))
            weights /= np.sum(weights)
            all_weights[n, :] = weights

            # calculate expected returns and volatility then save to arrays
            all_returns[n] =  np.dot(mu, weights) * 252
            all_volatility[n] = np.sqrt(np.dot(weights, np.dot(annual_cov, weights)))

            # calculate sharpe ratio to then save as well.
            all_sharpe[n] = all_returns[n] / all_volatility[n]
    
        
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # efficient frontier scatter plot colored with Sharpe Ratio
        scatter = ax.scatter(
            x=all_volatility, y=all_returns,
            c=all_sharpe,
            cmap="plasma",
            alpha=0.6
        )

        # --- get index & values of best max sharpe and lowest vol ---
        max_sr = all_sharpe.max()
        max_sr_idx = np.argmax(all_sharpe)

        # metrics and weights for optimized portfolio to display
        max_sr_weights = all_weights[max_sr_idx, :]
        max_sr_vol = all_volatility[max_sr_idx]
        max_sr_returns = all_returns[max_sr_idx]

        min_vol = all_volatility.min()
        min_vol_idx = np.argmin(all_volatility)
        
        # highlight with marker where the portfolio with max sharpe lays
        ax.scatter(
            x=all_volatility[max_sr_idx], y=all_returns[max_sr_idx], 
            marker="*", 
            color="red", 
            s=300,
            edgecolors='black',
            label=f"Max Sharpe: {max_sr:.3f}"
        )

        # highlight with marker where the portfolio with lowest vol lays
        ax.scatter(
            x=min_vol, y=all_returns[min_vol_idx], 
            color="blue", 
            edgecolors="black",
            s=150,
            label=f"Lowest Volatility: {min_vol:.3f}"
        )

        plt.colorbar(scatter, label="Sharpe Ratio")
        ax.set_xlabel("Annual Volatility (Risk)")
        ax.set_ylabel("Annual Expected Return (Gain)")
        ax.set_title("Efficient Frontier")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # create a data structure to store results
        MCPortfolio = namedtuple("MCPortfolio",
            ["all_weights", "all_returns", "all_volatility",
            "optim_weights", "optim_volatility", "optim_returns",
            "optim_sharpe"]
        )

        optimization_results = MCPortfolio(
            all_weights=all_weights,
            all_returns=all_returns,
            all_volatility=all_volatility,
            optim_weights=max_sr_weights,
            optim_volatility=max_sr_vol,
            optim_returns=max_sr_returns,
            optim_sharpe=max_sr,
        )

        if st:
            return fig, optimization_results 
        
        plt.show()
        return optimization_results
    
