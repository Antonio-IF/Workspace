# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore') 

# Function to extract data from Yahoo Finance
def get_yahoo_data(tickers, start_date, end_date):
    # Clean tickers by removing spaces
    tickers = [ticker.strip() for ticker in tickers]
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            raise ValueError("Could not obtain data for the specified tickers")
        
        # Check minimum observations for covariance calculation
        if len(data) < len(tickers) + 2:  # n+2 observations minimum for covariance
            raise ValueError("Insufficient data points for reliable covariance calculation")
        
        # Handle missing data by dropping completely empty rows
        data = data.dropna(how='all')
        if data.empty:
            raise ValueError("Data is empty after removing missing values")
        
        returns = data.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough data to calculate returns")
        
        # Use min_periods in cov() to ensure sufficient observations
        covariances = returns.cov(min_periods=len(tickers)+2)
        
        # Check if the covariance matrix contains NaN values
        if covariances.isnull().values.any():
            raise ValueError("Covariance matrix contains NaN values")
        
        return returns, covariances
    except Exception as e:
        raise ValueError(f"Error downloading data: {str(e)}")

# Function get minimum variance portfolio
def get_minimum_variance_portfolio_weights(returns, covariances):
    num_assets = len(returns.columns)
    
    # Add check for small matrices
    if num_assets < 2:
        raise ValueError("At least two assets are needed to create a portfolio")
    
    # Ensure covariances is a 2D matrix
    covariances = np.array(covariances)
    if len(covariances.shape) == 1:
        covariances = covariances.reshape(num_assets, num_assets)

    # Add matrix conditioning control
    try:
        inv_covariances = np.linalg.inv(covariances)
    except np.linalg.LinAlgError:
        # If the matrix is singular, use pseudoinverse
        inv_covariances = np.linalg.pinv(covariances)
    
    ones = np.ones(num_assets)
    weights_min_variance = np.dot(inv_covariances, ones) / np.dot(np.dot(ones, inv_covariances), ones)
    
    # Ensure weights sum to 1 and are non-negative
    weights_min_variance = np.maximum(weights_min_variance, 0)
    weights_min_variance = weights_min_variance / np.sum(weights_min_variance)

    return weights_min_variance

# Function of returns and variances
def calculate_return_and_variance(weights, returns, covariances):
    if len(weights) != returns.shape[1]:
        raise ValueError("Mismatch in number of assets and length of weights")

    portfolio_return = np.dot(weights, returns.mean())
    portfolio_variance = np.dot(np.dot(weights, covariances), weights)
    return portfolio_return, portfolio_variance

def calculate_efficient_frontier(returns, covariances, num_portfolios=50000):
    num_assets = len(returns.columns)
    
    # Check minimum observations for optimization
    if len(returns) < num_assets + 2:
        raise ValueError("Insufficient data points for reliable optimization")
    
    # Add check for few assets
    if num_assets < 2:
        raise ValueError("At least two assets are needed to calculate the efficient frontier")
    
    # Adjust number of portfolios for cases with few assets
    num_portfolios = min(num_portfolios, 50) if num_assets == 2 else num_portfolios
    
    returns_array = np.linspace(min(returns.mean()), max(returns.mean()), num_portfolios)
    efficient_portfolios = []
    
    for target_return in returns_array:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, returns.mean()) - target_return},
            {'type': 'ineq', 'fun': lambda x: x}  # Weights >= 0
        ]
        
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(covariances, x))), 
                        x0=np.array([1/num_assets]*num_assets),
                        method='SLSQP',
                        constraints=constraints)
        
        if result.success:
            efficient_portfolios.append({
                'return': target_return,
                'risk': np.sqrt(np.dot(result.x.T, np.dot(covariances, result.x))),
                'weights': result.x
            })
    
    return efficient_portfolios

def optimize_sharpe_ratio(returns, covariances, risk_free_rate=0.02):
    num_assets = len(returns.columns)
    
    def negative_sharpe(weights):
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(covariances * 252, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe  # Negative because minimize seeks the minimum
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weights = 1
        {'type': 'ineq', 'fun': lambda x: x}  # weights >= 0
    ]
    
    result = minimize(negative_sharpe,
                     x0=np.array([1/num_assets] * num_assets),
                     method='SLSQP',
                     constraints=constraints)
    
    return result.x

def perform_backtest_with_rebalancing(initial_investment, returns, rebalance_months=6, risk_free_rate=0.02):
    # Verify sufficient data for the first period
    if len(returns) < len(returns.columns) + 2:
        raise ValueError("Insufficient data points for reliable backtest")
    
    # Convert index to datetime if not already
    returns.index = pd.to_datetime(returns.index)
    
    # Initialize variables
    portfolio_values = []
    current_weights = None
    current_value = initial_investment
    weights_history = []
    dates_history = []
    
    # Define rebalancing dates
    start_date = returns.index[0]
    end_date = returns.index[-1]
    current_date = start_date
    
    while current_date <= end_date:
        # Define the period for optimization
        optimization_end = current_date
        optimization_start = returns.index[0]  # Use all available data up to current_date
        
        # Get data for optimization
        optimization_returns = returns[optimization_start:optimization_end]
        
        if len(optimization_returns) > 0:
            # Calculate covariances with historical data
            optimization_cov = optimization_returns.cov()
            
            # Optimize weights
            current_weights = optimize_sharpe_ratio(optimization_returns, optimization_cov, risk_free_rate)
            
            # Save weights and date
            weights_history.append(current_weights)
            dates_history.append(current_date)
        
        # Define simulation period (next 6 months)
        next_rebalance_date = current_date + pd.DateOffset(months=rebalance_months)
        simulation_period = returns[current_date:next_rebalance_date]
        
        if len(simulation_period) > 0:
            # Calculate portfolio returns for this period
            period_returns = np.dot(simulation_period, current_weights)
            
            # Modify the calculation of period_values to ensure correct lengths
            period_values = current_value * (1 + period_returns).cumprod()
            if len(portfolio_values) == 0:
                portfolio_values.extend(period_values.tolist())
            else:
                portfolio_values.extend(period_values.tolist()[1:])  # Avoid duplicating dates
            
            current_value = period_values[-1]
        
        # Move to the next rebalancing date
        current_date = next_rebalance_date
    
    # Ensure lengths match
    portfolio_values = pd.Series(portfolio_values[:len(returns.index)], index=returns.index[:len(portfolio_values)])
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] - initial_investment) / initial_investment * 100
    portfolio_returns = portfolio_values.pct_change().dropna()
    annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / volatility
    max_drawdown = np.min(portfolio_values/np.maximum.accumulate(portfolio_values) - 1)
    
    return {
        'portfolio_value': portfolio_values,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'weights_history': weights_history,
        'rebalance_dates': dates_history
    }

# Time period for data
start_date = '2020-11-01'
end_date = '2024-11-01'

# User input and data download section
while True:
    try:
        tickers = input("Enter stock tickers (comma-separated): ").split(',')
        returns, covariances = get_yahoo_data(tickers, start_date, end_date)
        if not returns.empty:
            break
    except ValueError as e:
        print(f"Error: {e}")
        print("Please try again with valid tickers (example: AAPL,MSFT)")
        continue

# Calculate the minimum variance portfolio
weights_min_variance = get_minimum_variance_portfolio_weights(returns, covariances)

# Print results
print("\nWeights of the minimum variance portfolio:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {weights_min_variance[i]:.4f}")

# Calculate the return and variance of the minimum variance portfolio
return_min_variance, variance_min_variance = calculate_return_and_variance(weights_min_variance, returns, covariances)

print("\nReturn of the minimum variance portfolio:", return_min_variance)
print("Variance of the minimum variance portfolio:", variance_min_variance)

# Calculate efficient frontier
efficient_portfolios = calculate_efficient_frontier(returns, covariances)
efficient_returns = [p['return'] for p in efficient_portfolios]
efficient_risks = [p['risk'] for p in efficient_portfolios]

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(efficient_risks, efficient_returns, c='blue', marker='o', label='Efficient Frontier')
plt.scatter(np.sqrt(variance_min_variance), return_min_variance, 
           color='red', marker='*', s=200, label='Minimum Variance Portfolio')

plt.title('Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()

# Perform backtest
initial_investment = 1_000_000  # $1 million
backtest_results = perform_backtest_with_rebalancing(initial_investment, returns)

# Print results
print("\nBacktesting Results (Rebalanced Portfolio):")
print(f"Initial Investment: ${initial_investment:,.2f}")
print(f"Final Portfolio Value: ${backtest_results['portfolio_value'][-1]:,.2f}")
print(f"Total Return: {backtest_results['total_return']:.2f}%")
print(f"Annualized Return: {backtest_results['annual_return']*100:.2f}%")
print(f"Annualized Volatility: {backtest_results['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {backtest_results['max_drawdown']*100:.2f}%")

# Print weight history
print("\nRebalancing History:")
for date, weights in zip(backtest_results['rebalance_dates'], backtest_results['weights_history']):
    print(f"\nRebalancing date: {date.strftime('%Y-%m-%d')}")
    for ticker, weight in zip(returns.columns, weights):
        print(f"{ticker}: {weight:.4f}")

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(backtest_results['portfolio_value'])
plt.title('Portfolio Value Over Time (With Rebalancing)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.grid(True)
plt.show()

# Get data from S&P 500 for the same period
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
sp500_returns = sp500_data.pct_change().dropna()

# Calculate value for S&P 500 portfolio
sp500_portfolio_value = initial_investment * (1 + sp500_returns).cumprod()

# Metrics for S&P 500
sp500_total_return = float((sp500_portfolio_value.iloc[-1] - initial_investment) / initial_investment * 100)
sp500_returns_series = sp500_portfolio_value.pct_change().dropna()
sp500_annual_return = float((1 + sp500_total_return/100) ** (252/len(sp500_returns)) - 1)
sp500_volatility = float(np.std(sp500_returns_series) * np.sqrt(252))
sp500_sharpe_ratio = float((sp500_annual_return - 0.02) / sp500_volatility)
sp500_max_drawdown = float(np.min(sp500_portfolio_value/np.maximum.accumulate(sp500_portfolio_value) - 1))

# Print Results for S&P 500
print("\nS&P 500 Results:")
print(f"Final Portfolio Value: ${float(sp500_portfolio_value.iloc[-1]):,.2f}")
print(f"Total Return: {sp500_total_return:.2f}%")
print(f"Annualized Return: {sp500_annual_return*100:.2f}%")
print(f"Annualized Volatility: {sp500_volatility*100:.2f}%")
print(f"Sharpe Ratio: {sp500_sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {sp500_max_drawdown*100:.2f}%")

# Comparative plots
plt.figure(figsize=(12, 6))
plt.plot(backtest_results['portfolio_value'], label='Optimized Portfolio')
plt.plot(sp500_portfolio_value, label='S&P 500')
plt.title('Comparison: Optimized Portfolio vs S&P 500')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.show()



