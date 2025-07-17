import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import entropy
from collections import namedtuple

class SharpeOptimization:
    def __init__(self, portfolio):
        """
        Initialize by getting all of the Portfolio's class
        attributes through composition
        """
        self.portfolio = portfolio

        # lazy checker to avoid recomputation and cached results 
        self._optimized = False
        self._optimization_results = None

    def sharpe_ratio(self, weights, negative_sharpe=False) -> float:
        """Calculate the Sharpe ratio with entropy regularization to let the user
        have a choice between the 
        
        Args:
            weights: List of stock weights to use
            negative_sharpe: If True, return negative Sharpe ratio for minimization
            
        Returns:
            Sharpe ratio (or negative Sharpe ratio if negative_sharpe is True)
        """
        # Calculate portfolio return with specified weights
        portfolio_return = np.dot(weights, self.portfolio.expected_returns_vector)
        
        # Calculate volatility (this should come from after using variance_and_volatility method)
        volatility = self.portfolio._volatility_cache
        
        # Annualized metrics
        annualized_return = portfolio_return * self.portfolio.trading_days
        annualized_vol = volatility * np.sqrt(self.portfolio.trading_days)
        
        # Sharpe ratio with entropy regularization
        sharpe = (annualized_return - self.portfolio.annual_rate) / (annualized_vol + 
                 self.portfolio.entropy_lambda * (1 - entropy(weights)/np.log(len(weights))))
        
        if negative_sharpe:
            return -sharpe
        
        return sharpe


    def get_optimized_portfolio(self, show_portfolio: bool=False) -> namedtuple:
        """Check to see flag if to avoid recomputation, if hasn't
        been done do optimization of portfolio via minimizing the negative
        sharpe ratio; 
        
        Returns:
            SharpePortfolio: Named tuple with optimization results
        """
        # Check if we've already optimized
        if self._optimized:
            return self._optimization_results
        
        # Make sure we have the data we need
        self.portfolio.expected_returns()

        initial_variance, initial_volatility = self.portfolio.variance_and_volatility()
        bounds, weights = self.portfolio.calculate_bounds()
        
        # Calculate initial sharpe ratio; just for demonstration inside final result
        initial_sharpe = self.sharpe_ratio(weights, False)
        
        # Run the optimization
        optimized_result = minimize(
            fun=self.sharpe_ratio,
            x0=weights,
            args=(True,),  # negative_sharpe=True
            bounds=bounds,
            method="SLSQP",
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "ineq", "fun": lambda w: w}
            ]
        )
        
        # Use optimized weights
        original_weights = self.portfolio.stock_weights
        self.portfolio.stock_weights = optimized_result.x
        
        # calculate brand the new expected return portfolio, variance and volatility based off optimized weight
        optimized_portfolio, new_total_expected_return = self.portfolio.expected_returns(
            want_expected_returns=show_portfolio,
            force_recalculation=True
        )
        new_variance, new_volatility = self.portfolio.variance_and_volatility(
            force_recalculation=True
        )
        
        # create the data structure to store results
        SharpePortfolio = namedtuple("SharpePortfolio", 
                                     ["portfolio", "expected_return", "variance", 
                                      "volatility", "initial_sharpe", "optimized_sharpe",
                                      "weights"])
        
        self._optimization_results = SharpePortfolio(
            portfolio=optimized_portfolio,
            expected_return=new_total_expected_return,
            variance=new_variance.item(),
            volatility=new_volatility.item(),
            initial_sharpe=initial_sharpe,

            # gets optimized sharpe from minimization object and weights
            optimized_sharpe=abs(optimized_result.fun),
            weights=optimized_result.x
        )
        
        # mark as optimized
        self._optimized = True
        
        # restore original weights to avoid side effects in overriding 
        self.portfolio.stock_weights = original_weights
        
        return self._optimization_results
    
    """Special methods to have optimized data structure act like python built-in data type"""

    def __repr__(self):
        return f"SharpeOptimization(num_stocks={self.portfolio.num_stocks!r}, start_date={self.portfolio.start_date!r}, end_date={self.portfolio.end_date!r})"

    def __str__(self):
        return f"{self._optimization_results}"
    
    # returns amount of sectors inside regular/optimized portfolio
    def __len__(self):
        return len(self._optimization_results)
    



