import numpy as np
import yfinance as yf
import cvxpy as cp

# Define tickers and get data
tickers = ["MSFT", "DE", "COST"]
data = yf.download(tickers, period="100d", auto_adjust=True)
prices = data["Close"]

# Define return matrix for three stocks
returns_matrix = ((prices/prices.shift(1)) - 1).dropna()

# Define risk tolerance
risk_tolerance = 0.015

# Define ambiguity radius
ambiguity_radius = 0.05

# Compute estimated return for each stock
estimated_return = returns_matrix.mean()

# Define parameters for optimization
T, n = returns_matrix.shape
mu = estimated_return.values
R_centered = (returns_matrix - mu).values

# Optimization variables
w = cp.Variable(n, nonneg=True)
u = cp.Variable(T) # Variable to represent the absolute value in the MAD definition

# MAD definition
MAD = (1/T) * cp.sum(u)

# Define Constraints
constraints = [
    cp.sum(w) == 1,
    u >= R_centered @ w,
    u >= -(R_centered @ w),
    MAD <= risk_tolerance
]

# Define objective function and optimization problem
objective_function = cp.Maximize(mu @ w - ambiguity_radius * MAD)
problem = cp.Problem(objective_function, constraints)
problem.solve()

# Compute portfolio's MAD with optimized weights
portfolio_deviation = R_centered @ w.value
optimized_MAD = np.mean(np.abs(portfolio_deviation))

# Compute portfolio's expected return with optimized weights
expected_portfolio_returns = mu @ w.value

print("MAD with optimized weights: ", optimized_MAD)
print("Expected portfolio returns: ",  expected_portfolio_returns)