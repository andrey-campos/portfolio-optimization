import copy
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from collections import namedtuple
from data_retriever import *
from decimal import Decimal, getcontext

# optimization strategies usable for the portfolio
from strategies.sharpe import SharpeOptimization
from strategies.mc_simulation import MCOptimization

class Portfolio():
    def __init__(self, num_stocks: int, start_date: str, end_date: str, # required parameters for user
        trading_days: int=254, annual_rate: float=0.0327, one_minus_lambda: float=0.04, # constants (can be changed)
        entropy_lambda: float=0.5, lower_bound: float=0.01, upper_bound: float=0.10, # optimization constants (can be changed)
        want_diversification: bool=True, precompute: bool=False, extracted_tickers=None, using_st=False # default bool constants 
        ) -> dict:

        """ 
        Initialize the Portfolio object with configuration parameters. Preprocess stock
        data and create interface that allows for max user customization for their portfolio
        
        #### Args:
            num_stocks: Number of stocks to select from each sector.
            start_date: Start date for historical data in format 'YYYY-MM-DD'.
            end_date: End date for historical data in format 'YYYY-MM-DD'.
            annual_rate: Risk-free rate for Sharpe ratio calculation.
            trading_days: Number of trading days in a year for annualization.
            one_minus_lambda: Smoothing factor for EWMA calculations (alpha).
            entropy_lambda: Weight for entropy regularization in optimization.
            upper_bound: Maximum weight allowed for any individual stock.
            lower_bound: Minimum weight allowed for any individual stock.
            want_diversification: allows user to choose if they want to diversify their portfolio or not
            preselected_data: allow user to use test/compact data for development purposes
            st: impersonates chrome session for streamlit to be able to download all stock data properly
            defaulted to false and downloads data regularly unless we use streamlit

            #### important
            precompute: allows user to precompute everything in instance variables
            this allows them to easily use portfolio methods to get cached pre-computed
            results

            if precompute=false:
                allows user to selectively perform portfolio methods
                to compute things like expected returns, volatility etc..
                does at the end of the day store all computed results into
                internal cached instance variables to avoid recomputation.
            
        #### Returns:
            Initialized portfolio object with weights.
        """

        # 1st constraint: set dictionary to hard-coded sectors 
        self.portfolio: dict = {
            "Finance": pd.DataFrame(),
            "Tech": pd.DataFrame(),
            "Industrial": pd.DataFrame(),
            "Energy": pd.DataFrame()
        }

        self.start_date = start_date
        self.end_date = end_date
        self.annual_rate = annual_rate
        self.trading_days = trading_days
        self.num_stocks = num_stocks
        self.total_expected_return: float = 0.0
        self.expected_returns_vector: list[float] = []
        self.lower_bounds = lower_bound
        self.upper_bounds = upper_bound
        self.one_minus_lambda = one_minus_lambda
        self.extracted_tickers = extracted_tickers
        self.using_st = using_st

        # control factor for how much information from stock for sharpe optimization
        self.entropy_lambda = entropy_lambda

        # initial stock weight as (1, n) vector with equal weight distribution 
        self.stock_weights: np.array =  np.full((1, len(self.portfolio)*self.num_stocks), (1 / (len(self.portfolio) * self.num_stocks))) 

        # always prepare data in initialization to avoid multiple downloads of yfinance data
        if self.extracted_tickers is None:
            sector_based_tickers: dict[list] = retrieve_tickers_compact_test()
            self.portfolio = self.prepare_data(sectors=sector_based_tickers, logarithmic_data=True)
        
        # should be same format as value returned from retrieve_tickers_compact_test except from our ticker extraction function
        else:
            sector_based_tickers: dict[list] = self.extracted_tickers
            self.portfolio = self.prepare_data(sectors=sector_based_tickers, logarithmic_data=True)
        
        # directly modify portfolio to be diversified before any computation
        if want_diversification:
            self.portfolio = self.diversify_sectors()
        
        # instance variables to cache computation results if user wants to precompute methods in constructor
        if precompute:
            self._precompute()
    
    def _precompute(self):
        """
        Precompute all metrics in correct order to avoid recomputation 
        """
        self._historical_returns_cache = self.historical_returns()
        self._expected_returns_cache, self._total_return_cache = self.expected_returns()
        self._variance_cache, self._volatility_cache = self.variance_and_volatility()
        self._bounds_cache, self._weights_cache = self.calculate_bounds()


    def prepare_data(self, sectors: dict, logarithmic_data=True) -> dict:   
        """
        Download and preprocess stock data from Yahoo Finance.
        
        ### Args:
            sectors: Dictionary with sector names as keys and lists of ticker symbols as values.
            logarithmic_data: If True, apply logarithmic transformation to stock prices for better statistical properties.
            
        ### Returns:
            Dictionary containing preprocessed stock data organized by sector.
        """

        # Create a new list containing each stock, where stocks comes from sectors.values(), and stock comes from each stocks list
        stocks = [stock for stocks in sectors.values() for stock in stocks]
        try:
            # replicate browser TLS fingerprint to avoid yf request limiting 
            # found solution: https://github.com/ranaroussi/yfinance/issues/2422
            if self.using_st:
                from curl_cffi import requests
                session = requests.Session(impersonate="chrome")
                stock_data = yf.download(
                    tickers=stocks, 
                    session=session, 
                    start=self.start_date, 
                    end=self.end_date, 
                    auto_adjust=True,
                    progress=True
                )["Close"]

                session.close()

            else:
                stock_data = yf.download(
                    tickers=stocks, 
                    start=self.start_date,
                    end=self.end_date, 
                    auto_adjust=True,
                    progress=True
                )["Close"]


            if (len(stocks) - len(stock_data.columns)) != 0:
                print(f"Missing stock data inside data")
                return stock_data 
            
            else:
                print(f"All stock data downloaded successfully")
    
                sector_mapping = {
                    "finance": "Finance",
                    "tech": "Tech",
                    "industrial": "Industrial",
                    "energy": "Energy"
                }
                
                # log all returns for better statistical properties
                for sector, stocks in sectors.items():
                    portfolio_sector = sector_mapping[sector]
                    sector_data = stock_data[stocks]

                    # Choose between log or regular data: default -> True
                    if logarithmic_data:
                        self.portfolio[portfolio_sector] = np.log(sector_data)
                    else:
                        self.portfolio[portfolio_sector] = sector_data

                    # filling any NaNs or Nas with values that come before it
                    self.portfolio[portfolio_sector] = self.portfolio[portfolio_sector].ffill()
                return self.portfolio
    
        except AttributeError as error: 
            print(f"Received {error} while retrieving historical stock data, please try again.")


    def diversify_sectors(self, 
            show_lowest_corr:bool=False, 
            display_diversification:bool=False
        ) -> dict:
        """
        Select stocks with lowest average correlation within each sector for diversification.
        
        ### Args:
            show_lowest_corr: If True, print the average correlation values for selected stocks.
            display_diversification: If True, print the newly diversified portfolio with removed stocks
            
        ### Returns:
            Dictionary containing diversified portfolio with low-correlation stocks.
        """
        getcontext().prec = 10 # for exact calculations
        diversified_assets = {}

        # make a correlation matrix per sector to use it to calculate average stock correlation
        for sector, data in self.portfolio.items():
            corr_matrix = data.corr()
            # use dictionary to save stock ticker and avg correlation
            avg_correlations = {}

            # outer loop allows us to loop through all row stock names per sector
            for stock in corr_matrix.index:
                correlations = [Decimal(str(x)) for other_stock, x in corr_matrix[stock].items() if other_stock != stock]
                avg_corr = sum(correlations) / Decimal(len(correlations))

                avg_correlations[stock] = float(avg_corr)

            # make all dictionary key value pairs to inside sectors to series's
            diversified_assets[sector] = pd.Series(avg_correlations)
            diversified_assets[sector] = diversified_assets[sector].nsmallest(self.num_stocks, "first")

        # Make a modified portfolio that removes the assets that AREN'T diversified:
        for dp_sector, dp_data in self.portfolio.items():
            dataframe_stocks, series_stocks = [], []
            dataframe_stocks = [col for col in self.portfolio[dp_sector].columns]

            for index in diversified_assets[dp_sector].index:
                series_stocks.append(index)
            # filters all stocks that are not inside the diversified stocks variable:
            self.portfolio[dp_sector] = self.portfolio[dp_sector].drop([stock for stock in dataframe_stocks if stock not in series_stocks], axis=1)
        
        # Allows to see the average correlation for the stocks that were picked
        if show_lowest_corr:
            print(f"\n\nDiversified portfolio stocks lowest correlation averages:\n{diversified_assets}")
            print(f"\nType of portfolio: {type(diversified_assets)}")
            print(f"\nType of data inside sectors keys: {type(diversified_assets["Finance"])}")

        if display_diversification:
            print(f"\nComplete and diversified stocks:\n{self.portfolio}")

        return self.portfolio


    def historical_returns(self, want_historical_returns: bool=False) -> dict:
        """
        Calculate historical/daily returns from price data using percentage change. Return this
        as a new data structure separate from our regular `self.portfolio` instance.

        ### Args:
            - want_historical_returns: defaulted to false, but prints out 
            historical_returns portfolio to console if true.

        ### Returns:
            Dictionary containing historical/daily returns for each stock.
        """

        # check if has been pre-computed; avoids repeated computations
        if hasattr(self, "_historical_returns_cache"):
            if want_historical_returns:
                print(f"\nHistorical Returns Portfolio:\n{self._historical_returns_cache}")

            return self._historical_returns_cache
        
        hist_returns_portfolio = {}

        # use pandas pct change to get period over period change
        for sector, data in self.portfolio.items():
            # percentage change calculates daily returns
            sector_returns = data.pct_change()

            # clean up any daily return calculations 
            sector_returns = sector_returns.dropna()
            sector_returns = sector_returns.replace([-np.inf, np.inf], 0)
            sector_returns = sector_returns.fillna(0)
            
            # finally add sector: sector_returns relationship to dictionary
            hist_returns_portfolio[sector] = sector_returns
    
        
        if want_historical_returns:
            print(f"\nHistorical Returns Portfolio:\n{hist_returns_portfolio}")

        # finalize results into instance variables; prevents re-computing
        self._historical_returns_cache = hist_returns_portfolio

        return hist_returns_portfolio
        

    def expected_returns(self, want_expected_returns: bool=False, force_recalculation: bool=False) -> Tuple[dict, float]:
        """
        Calculate expected returns using Exponentially Weighted Moving Average (EWMA). Use this operation to
        create an entirely new formatted portfolio based on these expected returns, 
        adding weights per stock and useful statistics for the portfolio.
            
        ### Returns:
            Dictionary containing expected returns and stock weights for the portfolio.
            Float of the portfolio's total expected return
            If want_expected_returns is True, also prints the portfolio's total expected return.
        """
        
        # skip cache if force_recalculation is true
        if not force_recalculation and hasattr(self, "_expected_returns_cache"):
            # if user wants to see the portfolio's actual expected return; return the analyzed portfolio & portfolio expected return
            if want_expected_returns:
                print(f"\nThe portfolio's total expected return based on historical return's: {self.total_expected_return:2f}\n")
                print(f"\nThe expected returns portfolio:\n{self._expected_returns_cache}")
        
            return self._expected_returns_cache, self._total_return_cache


        # make new variables to store unique portfolio types built-off original portfolio
        historical_returns_portfolio = self.historical_returns()

        # avoids deep copy just by replication of the self.portfolio structure
        expected_returns_portfolio = {}  # Start with empty dict to avoid nested structure issues
        
        # Initialize sectors in expected_returns_portfolio to match our original portfolio structure
        for sector in self.portfolio.keys():
            expected_returns_portfolio[sector] = pd.DataFrame()

        weight_index = 0 

        # flatten just incase its (4, 5), num of sector's * num of stocks in sectors , 1 / num of sectors, and stock tickers to save all organized stock tickers
        self.stock_weights = self.stock_weights.flatten()
        self.sector_weight = 1/len(historical_returns_portfolio)
        self.stock_tickers = []
       
         # Reset total expected return before calculation
        self.total_expected_return = 0.0
        self.expected_returns_vector = []

        for sector, data in historical_returns_portfolio.items():
            # reset list to allow us to add correct stock names per sector
            stock_names = []
            # turn this dictionary into a new pd dataframe eventually
            expected_returns_portfolio[sector] = {
                "Expected_Returns": [],
                "Stock_Percentage": [],
            }
            
            for series_name, series in data.items(): 
                # calculate stock's expected return using pandas ewm method and mean to get EWMA, get last element to represent EWMA
                stock_expected_return = (series.ewm(alpha=self.one_minus_lambda).mean()).iloc[-1]
        
                # use this vector that contains each individual stock expected return for the optimization
                self.expected_returns_vector.append(stock_expected_return)

                # while other for actual use and metrics
                stock_names.append(series_name)
                self.stock_tickers.append(series_name)

                # append most recent expected return from EWMA algorithm & the stock weight
                expected_returns_portfolio[sector]["Stock_Percentage"].append(self.stock_weights[weight_index])
                weight_index += 1
                expected_returns_portfolio[sector]["Expected_Returns"].append(stock_expected_return)


            # when all data is finished being added, turn it to a dataframe and have indexes be tickers
            expected_returns_portfolio[sector] = pd.DataFrame(expected_returns_portfolio[sector], index=stock_names)


        # look for expected return's column, then use this column calculate each column's expected return by iteratively adding to it per sector
        for sector, data in expected_returns_portfolio.items():
            # turn both columns in to numpy arrays to allow to get dot product for sector's expected return
            expected_returns = np.array(data["Expected_Returns"])
            sector_weight = np.array(data["Stock_Percentage"])
            sector_expected_return = np.dot(sector_weight, expected_returns)

            # incremently add it so get a portfolio expected return 
            self.total_expected_return += sector_expected_return.item()

        # if user wants to see the portfolio's actual expected return; return the analyzed portfolio & portfolio expected return
        if want_expected_returns:
            print(f"\nThe portfolio's total expected return based on historical return's: {self.total_expected_return:2f}\n")
            print(f"\nThe expected returns portfolio:\n{expected_returns_portfolio}")
        
        # finalize results into instance variables; prevents re-computing
        self._expected_returns_cache = expected_returns_portfolio
        self._total_return_cache = self.total_expected_return

        return expected_returns_portfolio, self.total_expected_return
    

    def variance_and_volatility(self, force_recalculation: bool=False) -> Tuple[np.array, np.array]:
        """
        calculate portfolio variance and volatility based on the covariance matrix from daily returns

        ### Args:
            - force_recalculation: defaulted to false, but recalculates variance and volatility
            if user wants to avoid using pre-computed results if true.

        ### Returns:
            A tuple containing portfolio variance and volatility (standard deviation).
        """
        
        # skip cache if force_recalculation is true
        if not force_recalculation and hasattr(self, "_variance_cache") and hasattr(self, "_volatility_cache"):
            return self._variance_cache, self._volatility_cache

        hist_returns_portfolio = self.historical_returns()

        combined_returns = pd.concat(
            [hist_returns_portfolio[sector] for sector in hist_returns_portfolio],
            axis=1              
        )

        # make temporary 1d weights array for simpler calculations 
        weights = self.stock_weights.flatten()

        cov_matrix = combined_returns.cov()
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        # finalize results into instance variables; prevents re-computing
        self._variance_cache = portfolio_variance
        self._volatility_cache = np.sqrt(portfolio_variance)

        # square root to return both variance and volatility
        return self._variance_cache, self._volatility_cache
        

    
    def calculate_bounds(self, force_recalculation: bool=False) -> Tuple[tuple, list[float]]:
        """
        Create bounds for optimization to constrain stock weights within specified limits by stacking 
        all returns dataframe in expected returns data structure vertically and getting all weights 
        into a single column.

        ### Args:
            - force_recalculation: defaulted to false but recalculates new bounds and weights if true 

        ### Returns:
            A tuple containing bounds for optimization and the current list of stock weights.
        """

        if not force_recalculation and hasattr(self, "_bounds_cache") and hasattr(self, "_stock_weight_cache"):
            return self._bounds, self._stock_weight

        # only get first element which is just portfolio as a dictionary
        expected_returns_portfolio, _ = self.expected_returns()

        # turn all sectors dataframes into huge dataframe to get all stock weights
        return_dataframe = pd.concat([expected_returns_portfolio[keys] for keys in expected_returns_portfolio.keys()])
        weights = return_dataframe["Stock_Percentage"].to_list()

        # use tuple comprehension to make bounds for scipy minimize, also choose maximum value for weight
        bounds = tuple((self.lower_bounds, self.upper_bounds) for weight in range(len(weights)))

        # finalize results into instance variables; prevents re-computing
        self._bounds = bounds
        self._stock_weight = weights

        return self._bounds, self._stock_weight
    
    
    def process_portfolio(self, display_portfolios: bool=False) -> namedtuple:
        """
        Function to perform all portfolio methods and store
        the results of said methods into lightweight 'PortfolioResult' data structure. 
        Serves as alternative to using `Portfolio` object and saved
        variables / performing methods one at a time.

        ### Args:
            display_portfolios: if True, allows user to print both historical
            and expected return portfolios to terminal

        ### Returns:
            Named tuple called 'PortfolioResult' that allows easy referencing of 
            attributes through method notation, for example: 
            `PortfolioResult.variance` -> variance
        """

        PortfolioResult = namedtuple("PortfolioResult", 
                        ["portfolio", "historical_returns", "portfolio_return", 
                        "expected_returns", "variance","volatility", "bounds", 
                        "weights"])

        # prepares both data and diversifies sectors
        self.portfolio = self.diversify_sectors()

        if display_portfolios:
            historical_returns_portfolio = self.historical_returns(want_historical_returns=True)
            expected_returns_portfolio, total_expected_return = self.expected_returns(want_expected_returns=True)

        # perform all Portfolio methods to return to user
        historical_returns_portfolio = self.historical_returns()
        expected_returns_portfolio, total_expected_return = self.expected_returns()
        variance, volatility = self.variance_and_volatility()
        bounds, weights = self.calculate_bounds()

        # return small data structure of portfolio results
        return PortfolioResult(
            portfolio=self.portfolio,
            historical_returns=historical_returns_portfolio,
            expected_returns=expected_returns_portfolio,
            portfolio_return=total_expected_return,
            variance=variance,
            volatility=volatility,
            bounds=bounds,
            weights=weights
        )


    # --- Portfolio visualization methods ---

    def asset_piechart(self, st: bool=False, show_one_sector: bool=False, 
                       input_portfolio: dict=None, sector: bool=None,
                        display_optim_weights=False, optim_weights=None
        ) -> plt:
        """
        ## Visualize asset allocation with pie charts.
        
        ### Args:
            show_one_sector: If True, show allocation for a specific sector.
            st: Automatically false and is used as a switch to determine if function is used for streamlit or not
            input_portfolio: Portfolio (regular, expected, returns) data to visualize.
            sector: Specific sector to visualize if show_one_sector is True.
            
        ### Returns:
            Matplotlib plot object with pie chart visualization (fig variable) for streamlit
        """

        if input_portfolio is None:
            input_portfolio, _ = self.expected_returns()
        
        if show_one_sector:
            sector_tickers = [index for index in input_portfolio[sector].index]
            sector_weights = (input_portfolio[sector]["Stock_Percentage"]).tolist()
            fig, ax = plt.subplots()
            ax.set_title(f"Asset allocation for {sector}:", fontsize=15)
            ax.pie(sector_weights, labels=sector_tickers)

        else:
           # extract weights from input_portfolio instead of self.stock_weights (prev bug)
            all_tickers = []
            all_weights = []
            
            # het tickers and weights from the input portfolio
            for sector_name, sector_data in input_portfolio.items():
                all_tickers.extend(list(sector_data.index))
                all_weights.extend(sector_data["Stock_Percentage"].tolist())

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_title("Asset allocation for all stock tickers", fontsize=15)

            if display_optim_weights == False:
                ax.pie(all_weights, labels=all_tickers, autopct='%1.1f%%')
            else:
                ax.pie(optim_weights, labels=all_tickers, autopct='%1.1f%%')

        if st:
            return fig
        else:
            plt.show()
            return fig 
    
    
    def correlation_heatmap(self, input_portfolio: dict=None) -> plt:
        """
        ## Create correlation heatmaps for stocks in each sector.
        
        ### Args:
            input_portfolio: Portfolio (regular, expected, returns) data to analyze correlations.

            
        ### Returns:
            Matplotlib plot object with correlation heatmaps
        """
        if input_portfolio is None:
            input_portfolio = self.portfolio
            
        for sector, data in input_portfolio.items():
            sns.heatmap(data.corr(numeric_only=True), cmap="YlGnBu", annot=True)
            plt.title(f"{sector}'s correlation heatmap:", fontsize=15)
            plt.show()
    

    def correlation_heatmap_all_sectors(self, input_portfolio: dict=None) -> plt:
        """
        ## Create correlation heatmap for all sectors to display in Streamlit UI

        ### Args:
            input_portfolio: Portfolio (regular, expected, returns) data to analyze correlations.
        
        ### Returns:
            Matplotlib plot object with correlation heatmaps (fig variable) for streamlit
        """
         # Combine all sector data into one DataFrame
        if input_portfolio is None:
            input_portfolio = self.portfolio

        all_data = pd.concat([data for data in input_portfolio.values()], axis=1)
        
        # Create single heatmap
        fig, ax = plt.subplots(figsize=(12, 11.68))
        sns.heatmap(all_data.corr(numeric_only=True), 
                    cmap="YlGnBu", 
                    annot=True, 
                    ax=ax,
                    fmt='.2f',
                    square=True)
        ax.set_title("Portfolio Correlation Heatmap (All Sectors)", fontsize=14)
        plt.tight_layout()
        
        return fig


    # --- Special methods to have data structure act like python built-in data type ---
    def __len__(self):
        return len(self.portfolio) # num of sectors 

    def __str__(self):
        return f"\nUser Portfolio:\n{self.portfolio!r}"

    def __repr__(self):
        return f"Portfolio(num_stocks={self.num_stocks!r}, start_date={self.start_date!r}, end_date={self.end_date!r})"
    

# --- main script example usage ---
if __name__ == "__main__":
    import time
    
    print("=== PERFORMANCE TEST ===")
    start_time = time.time()
    
    # Initialize portfolio (this will still take some time for data download)
    print("1. Initializing portfolio...")
    init_start = time.time()
    portfolio = Portfolio(
        num_stocks=2, 
        start_date="2022-08-08", 
        end_date="2023-08-12", 
        precompute=False  # Don't precompute everything
    )
    init_time = time.time() - init_start
    print(f"    Portfolio initialization: {init_time:.2f} seconds")
    
    # Only compute what's needed for efficient frontier
    print("2. Setting up MC optimizer...")
    mc_start = time.time()
    mc_optimizer = MCOptimization(portfolio)
    
    print("3. Running efficient frontier...")
    mc_optimizer.efficient_frontier(n_portfolios=1000)
    
    mc_time = time.time() - mc_start
    total_time = time.time() - start_time
    
    print(f"   MC setup + efficient frontier: {mc_time:.2f} seconds")
    print(f"   Total runtime: {total_time:.2f} seconds")

    # example usage with sharpe optimization
    # sharpe_optimizer = SharpeOptimization(portfolio)
    # results = sharpe_optimizer.get_optimized_portfolio()

    # print(f"\nComplete Optimized Portfolio:\n{sharpe_optimizer}")
    # print(f"Optimized Portfolio new sharpe ratio: {results.optimized_sharpe}")





