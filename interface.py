import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.colors as pc
import plotly.graph_objects as go

from portfolio import Portfolio
from data_retriever import *
from datetime import datetime, timedelta
from strategies.sharpe import SharpeOptimization
from strategies.mc_simulation import MCOptimization


# --- Page config and styling ---
st.set_page_config(
    page_title="portfolio-optimizer",
    page_icon="📊",
    layout="wide"
    
)

st.sidebar.success("View a detailed explanation of the algorithms below")

st.sidebar.title("Sharpe Ratio Optimization")
st.sidebar.write("""
    - Sharpe Ratio Maximization\n
    - Sector Based Asset Diversification\n
    - Entropy Regularization\n
""")

st.sidebar.title("Monte Carlo Optimization")
st.sidebar.write("""
    - Monte Carlo Simulation of Portfolio\n
    - Portfolio Plotted Against Efficient Frontier\n
    - VaR and CVaR Calculations\n
""")

st.sidebar.title("Black-Scholes Merton Options Pricer (Soon)")
st.sidebar.write("""
    - Put and Call Option Pricing
    - Put-Call Parity Calculations
    - Visualization of the Effects of Volatility  
""")


# --- CSS for the page and helper functions related to plot styling ---
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }

    .stTitle {
        font-weight: bold;
        color: #0f2537;
    }
    .tool-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


def sector_color_palette() -> dict[str]:
    """
    Get colors that scale with sectors, not individual stocks
    """
    base_colors: list[str] = px.colors.qualitative.Pastel 

    # assign each sector a hex-code color
    return {
        "Finance": base_colors[0],
        "Tech": base_colors[1],
        "Industrial": base_colors[2],
        "Energy": base_colors[3],
    }


# --- Main elements of the page used for displaying portfolio optimization results ---

# header and intro to page
st.title("📊 Welcome to the Portfolio Optimization Engine.")
st.markdown("### Enter a csv of tickers you want to use for your portfolio " \
"so we can extract tickers for your optimization strategy!")

# --- ticker extraction tool: give user option between testing data or their own CSV formatted data ---
genre = st.radio(
    "Choose your form of data collection for your portfolio's stocks..",
    ["CSV Stock Ticker Extraction", "Preselected Stock Tickers"],
    captions=[
        f"Upload a CSV with the format of: sector,ticker,*",
        "Developer/Testing mode of platform with preselected stock tickers to reduce overhead."
    ]

)
if genre == "CSV Stock Ticker Extraction":
    tickers_csv = st.file_uploader("Upload your ticker data here! (CSV format only).")
    try:
        sector_tickers = retrieve_tickers_streamlit(tickers_csv)

    # error validation from module function
    except Exception as e:
        st.error(f"Could not continue with uploaded file: {str(e)}")

else:
    sector_tickers = None
    st.info("Please be aware all data downloaded is preselected.")


        
# user enters interval of data
st.markdown("### Enter interval of stock data you want to use for your stocks.")

# the dates to enter inside portfolio instance user will create (must be str)
start_date: str = st.date_input("Start Date").strftime("%Y-%m-%d")
end_date: str = st.date_input("End Date").strftime("%Y-%m-%d")


# user enters strategy and number of stocks they want to optimize for 
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Enter your type of portfolio optimization strategy (only supports two methods..)")
    user_strategy = st.selectbox("Pick one", ["Sharpe Maximization", "Monte Carlo Simulation"])

with col2:
    st.markdown("### Enter number of stocks you want for portfolio diversification")
    user_num_stocks = st.number_input("Pick a number (1, 10)", 1, 10)

# UI for extra inputs needed for user to perform the MC simulation
if user_strategy == "Monte Carlo Simulation":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        timeframe = st.number_input("Pick a numbers of days for simulation (1, 10000)", 1, 10000)
    with col2:
        n_sims = st.number_input("Pick a numbers of simulations  (1, 10000)", 1, 10000)
    with col3:
        init_value = st.number_input("Pick a initial portfolio value (1, 100000)", 1, 100000)
    with col4:
        n_portfolios = st.number_input("Pick the amount of portfolios (1, 10000)", 1, 10000)


# --- styling / padding for 'market overview' section ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 5rem;    
        }
    </style>
""", 
unsafe_allow_html=True)

# show user useful metrics and results of portfolio optimization
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🪙 Market Overview For Selected Tickers")
with col2:
    optimized = st.button("Optimize Portfolio")

# initialize session state for portfolio
if "portfolio" not in st.session_state:
    st.session_state.portfolio = None
    st.session_state.portfolio_created = False    

# start portfolio instance and download data if user wants to optimize
if optimized and not st.session_state.portfolio_created:

    # show loading signal for user and display dates of downloaded stocks 
    with st.spinner("Downloading stock data and optimizing portfolio.."):
        st.info(f"Downloading data from {start_date} to {end_date}")

        try:
            portfolio = Portfolio(
                num_stocks=user_num_stocks, 
                start_date=start_date, 
                end_date=end_date,
                precompute=False, 

                # is none if user wants preselected stock tickers: mostly used for testing..
                extracted_tickers=sector_tickers,
                using_st=True
            )
            
            # check if portfolio has stock data downloaded 
            has_data = any(not df.empty for df in portfolio.portfolio.values())

            if has_data:
                st.session_state.portfolio = portfolio
                st.session_state.portfolio_created = True
                st.success("Portfolio created successfully.")
                
            else:
                raise Exception

        except (Exception, ValueError) as e:
            st.error(f"Couldn't create portfolio: {str(e)}")
            st.info(f"This is likely due to the rate limit for downloading data.. Waiting lasts around 1-5 minutes.")


# get portfolio instance from memory -> check if it has been made -> get attributes from instance to display
if st.session_state.portfolio_created and st.session_state.portfolio:
    portfolio = st.session_state.portfolio 
    
    # unoptimized portfolio statistics (purely for comparison)
    var, vol = portfolio.variance_and_volatility()
    var = float(var.item())
    vol = float(vol.item())
    _, exp_returns = portfolio.expected_returns()
    init_weights = portfolio.stock_weights

    # initialize variables with default values
    new_var = None
    new_vol = None 
    new_exp_returns = None
    init_sharpe = None
    optim_sharpe = None
    optimization_completed = False
    optim_weights = None

    if user_strategy == "Sharpe Maximization":
        try: 
            with st.spinner("Optimizing Portfolio"):
                sharpe_optimizer = SharpeOptimization(portfolio)
                optim_portfolio = sharpe_optimizer.get_optimized_portfolio()

                # optimized portfolio statistics 
                new_portfolio=optim_portfolio.portfolio
                new_var = optim_portfolio.variance
                new_vol = optim_portfolio.volatility 
                new_exp_returns = optim_portfolio.expected_return
                init_sharpe = optim_portfolio.initial_sharpe
                init_sharpe = float(init_sharpe.item())
                optim_sharpe = optim_portfolio.optimized_sharpe
                optim_weights = optim_portfolio.weights

                optimization_completed = True
                row_names = new_portfolio["Finance"].index

        except Exception as e:
            st.error(f"Optimization Failed: {str(e)}")

    # display the future paths and the efficient frontier plot alongside the metrics plot..
    elif user_strategy == "Monte Carlo Simulation":
        mc_optimizer = MCOptimization(portfolio)

        st.markdown("## Portfolio Future Paths and the Efficient Frontier")
        col1, col2 = st.columns(2)

        # do optimization through mc instance methods and display through plots
        with col1:
            st.pyplot(mc_optimizer.mc_simulation(
                    timeframe=timeframe,
                    n_sims=n_sims,
                    init_portfolio_value=init_value,
                    st=True
            ))   
        with col2:
            # unpack tuple that gives fig for streamlit and optimized results in named tuple
            fig, optim_results = mc_optimizer.efficient_frontier(
                n_portfolios=n_portfolios,
                st=True
            )
            st.pyplot(fig)

            # get optimized metrics from efficient frontier run 
            new_var = (optim_results.optim_volatility)**2
            new_vol = optim_results.optim_volatility 
            new_exp_returns = optim_results.optim_returns
            optim_sharpe = optim_results.optim_sharpe 
            optim_weights = optim_results.optim_weights

            optimization_completed = True



    else:
        st.warning("You must select a strategy to see the metrics of a optimized portfolio!")
    

    # --- Regardless of strategy weights are optimized ---
    portfolio.stock_weights = optim_weights
    # --- Instance weights should be switched for asset allocation chart function ----


    # --- Plots from plotly that displays useful information based on the portfolio output ---
    col1, col2 = st.columns(2)
    sector_colors = sector_color_palette()

    # multi-sector price movement chart using actual portfolio data
    with col1:
        fig = go.Figure()  

        for sector, data in portfolio.portfolio.items():
            
            normalized_prices = pd.DataFrame()

            for stock in data.columns:
                stock_price: list[float] = np.exp(data[stock])

                # normalize to start at 100 
                normalized_stock: list[float] = (stock_price / stock_price.iloc[0]) * 100
                normalized_prices[stock] = normalized_stock
            
            # calcs avg and does so horizontally so each stock is added by day -> then averaged to get an entry
            sector_average: list[float] = normalized_prices.mean(axis=1)

            # add sector weight figure to overall plot
            base_color = sector_colors[sector]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=sector_average,
                    mode="lines",
                    name=f"{sector} sector",
                    line=dict(color=base_color, width=3),
                    hovertemplate=f"<b>{sector} Sector</b><br>" +
                    "Date: %{x}<br>" +
                    "Index Value: %{y:.2f}<br>" + 
                    "<extra></extra>"
                )
            )
        
        fig.update_layout(
            title="Multi-Sector Percentage Performance",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # risk-return scatter plot
    with col2:
        fig = go.Figure()
        
        # Get expected returns and calculate individual stock volatilities
        expected_returns_portfolio, _ = portfolio.expected_returns()
        historical_returns = portfolio.historical_returns()
        
        for sector, data in expected_returns_portfolio.items():
            # Get stock returns and volatilities for this sector
            sector_historical = historical_returns[sector]
            
            for stock in data.index:
                expected_return = data.loc[stock, 'Expected_Returns']
                stock_volatility = sector_historical[stock].std() * np.sqrt(portfolio.trading_days)  # Annualized volatility
                weight = data.loc[stock, 'Stock_Percentage']
                
                fig.add_trace(go.Scatter(
                    x=[stock_volatility],
                    y=[expected_return],
                    mode='markers',
                    name=f'{stock}',
                    marker=dict(
                        size=weight * 1000,  # size based on weight
                        color=sector_colors.get(sector, '#7f7f7f'),
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f'<b>{stock}</b><br>' +
                                'Risk (Vol): %{x:.4f}<br>' +
                                'Return: %{y:.4f}<br>' +
                                f'Weight: {weight:.2%}<br>' +
                                f'Sector: {sector}<extra></extra>',
                    showlegend=False
                ))
        
        # add sector legend manually
        for sector, color in sector_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=sector,
                showlegend=True
            ))
        
        fig.update_layout(
            title="Risk-Return Profile by Stock",
            xaxis_title="Risk (Annualized Volatility)",
            yaxis_title="Expected Return",
            height=400,
            hovermode='closest',
            annotations=[
                dict(
                    text="Bubble size = Portfolio Weight",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)


    # --- Metrics Section: Old/New Expected Returns, Historical Returns, Volatility, and Variance ---
    st.markdown("## 📉 Unoptimized Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Expected Returns",
            value=f"{exp_returns:.4f}" if exp_returns is not None else "N/A",
        )

    with col2:
        st.metric(
            label="Portfolio Volatility", 
            value=f"{vol:.4f}" if vol is not None else "N/A",
        )

    with col3:
        st.metric(
            label="Portfolio Variance",
            value=f"{var:.4f}" if var is not None else "N/A",
        )

    with col4:
        st.metric(
            label="Initial Sharpe Ratio",
            value=f"{init_sharpe:.4f}" if init_sharpe is not None else "N/A",
        )
    
    # Only show optimized metrics if optimization was completed
    if optimization_completed:
        st.markdown("## 📈 Optimized Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Optimized Expected Returns",
                value=f"{new_exp_returns:.4f}" if new_exp_returns is not None else "N/A",
                delta=f"{(new_exp_returns - exp_returns):.4f}" if (new_exp_returns and exp_returns) else None
            )

        with col2:
            st.metric(
                label="Optimized Portfolio Volatility",
                value=f"{float(new_vol):.4f}" if new_vol is not None else "N/A",
                delta=f"{(new_vol - vol):.4f}" if (new_vol is not None and vol is not None) else None
            )

        with col3:
            st.metric(
                label="Optimized Portfolio Variance", 
                value=f"{float(new_var):.6f}" if new_var is not None else "N/A",
                delta=f"{(new_var - var):.6f}" if (new_var is not None and var is not None) else None
            )

        with col4:
            st.metric(
                label="Optimized Sharpe Ratio",
                value=f"{optim_sharpe:.4f}" if optim_sharpe is not None else "N/A",
                delta=f"{(optim_sharpe - init_sharpe):.4f}" if (optim_sharpe and init_sharpe) else None
            )


        # columns to display the pie chart and correlation heatmap of the portfolio 
        st.markdown("## 📊 Visual Metrics of the Optimized Portfolio")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(portfolio.asset_piechart(
                st=True, 
                display_optim_weights=True, 
                optim_weights=optim_weights
            ))

        with col2:
            st.pyplot(portfolio.correlation_heatmap_all_sectors())

        st.info("Charts will not change if inputs modified: Please reload screen.")



# --- Call to Action / Footer Section ---
st.markdown("---")

st.markdown("### Help with Our Portfolio Optimization Tools!")
st.markdown("""
    Do you want to help in making professional-grade portfolio optimization tools?
    Contribute at https://github.com/andrey-campos/portfolio-optimization to 
    progress our quantitative toolset!
""")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made by Andrey Campos.
    </div>
""", unsafe_allow_html=True)

