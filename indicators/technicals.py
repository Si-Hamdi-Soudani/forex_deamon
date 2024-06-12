import numpy as np
import pandas as pd

def calculate_sma(price_data, window):
  """Calculates the Simple Moving Average (SMA).

  Args:
    price_data: A list of tuples (timestamp, price).
    window: The SMA window size (number of periods).

  Returns:
    A list of SMA values.
  """
  if len(price_data) < window:
    return []  # Not enough data for SMA calculation

  prices = np.array([p[1] for p in price_data])  # Extract prices
  sma_values = np.convolve(prices, np.ones(window), 'valid') / window
  return sma_values.tolist()

def calculate_ema(price_data, window):
  """Calculates the Exponential Moving Average (EMA).

  Args:
    price_data: A list of tuples (timestamp, price).
    window: The EMA window size (number of periods).

  Returns:
    A list of EMA values.
  """
  if len(price_data) < window:
    return []  # Not enough data for EMA calculation

  prices = np.array([p[1] for p in price_data])
  alpha = 2 / (window + 1)
  ema_values = [prices[0]]  # Initialize with the first price

  for i in range(1, len(prices)):
    ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
    ema_values.append(ema)

  return ema_values

def calculate_rsi(price_data, window):
  """Calculates the Relative Strength Index (RSI).

  Args:
    price_data: A list of tuples (timestamp, price).
    window: The RSI window size (number of periods).

  Returns:
    A list of RSI values.
  """
  if len(price_data) < window + 1:
    return []  # Not enough data for RSI calculation

  prices = np.array([p[1] for p in price_data])
  price_changes = np.diff(prices)

  gains = np.where(price_changes > 0, price_changes, 0)
  losses = np.where(price_changes < 0, -price_changes, 0)

  avg_gain = np.convolve(gains, np.ones(window), 'valid') / window
  avg_loss = np.convolve(losses, np.ones(window), 'valid') / window

  rs = avg_gain / avg_loss
  rsi_values = 100 - (100 / (1 + rs))
  return rsi_values.tolist()



def calculate_macd(price_data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    prices = np.array([p[1] for p in price_data])
    df = pd.DataFrame({'Price': prices})
    df['EMA_Fast'] = df['Price'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Price'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df['MACD'].values.tolist(), df['Signal'].values.tolist()

def calculate_bollinger_bands(price_data, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    prices = np.array([p[1] for p in price_data])
    df = pd.DataFrame({'Price': prices})
    df['SMA'] = df['Price'].rolling(window=period).mean()
    df['StdDev'] = df['Price'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + df['StdDev'] * std_dev
    df['Lower'] = df['SMA'] - df['StdDev'] * std_dev
    return df['Upper'].values.tolist(), df['Lower'].values.tolist()
