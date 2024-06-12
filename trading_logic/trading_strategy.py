from indicators import technicals
from trading_logic import pattern_recognition
import os
import pickle
import time

class TradingStrategy:
    """Represents a trading strategy with parameters and evaluation methods."""

    def __init__(self, name, conditions, actions, parameters=None, strategy_id=None):
        self.strategy_id = strategy_id or str(int(time.time() * 1000))  # Unique ID
        self.name = name
        self.conditions = conditions  # List of condition functions
        self.actions = actions  # List of corresponding actions
        self.parameters = parameters or {}  # Dictionary of parameters
        self.performance = {"wins": 0, "losses": 0}

    def evaluate(self, price_data):
        """Evaluates the strategy on historical price data."""
        for condition, action in zip(self.conditions, self.actions):
            if condition(price_data, **self.parameters):
                return {"action": action}
        return None

    def update_performance(self, result):
        """Updates the strategy's performance based on trade outcomes."""
        if result == "win":
            self.performance["wins"] += 1
        elif result == "loss":
            self.performance["losses"] += 1

    def get_win_rate(self):
        """Calculates the win rate of the strategy."""
        total_trades = self.performance["wins"] + self.performance["losses"]
        if total_trades > 0:
            return self.performance["wins"] / total_trades
        else:
            return 0

    def save(self, data_dir="data"):
        """Saves the trading strategy to a file."""
        strategy_file = os.path.join(data_dir, f"strategy_{self.strategy_id}.pkl")
        os.makedirs(data_dir, exist_ok=True)
        with open(strategy_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(strategy_id, data_dir="data"):
        """Loads a trading strategy from a file."""
        strategy_file = os.path.join(data_dir, f"strategy_{strategy_id}.pkl")
        try:
            with open(strategy_file, "rb") as f:
                loaded_strategy = pickle.load(f)

                # Ensure loaded strategy has conditions and actions
                if not hasattr(loaded_strategy, 'conditions') or not hasattr(loaded_strategy, 'actions'):
                    raise ValueError("Loaded strategy missing conditions or actions")

                return loaded_strategy
        except (FileNotFoundError, ValueError, EOFError) as e:  # Catch EOFError
            print(f"Error loading strategy from {strategy_file}: {e}")
            return None

# --- Example Strategies ---

def sma_crossover_condition(price_data, short_sma_window, long_sma_window):
    """Condition for SMA crossover strategy."""
    short_sma = technicals.calculate_sma(price_data, short_sma_window)
    long_sma = technicals.calculate_sma(price_data, long_sma_window)
    return short_sma[-1] > long_sma[-1] and short_sma[-2] <= long_sma[-2]

def rsi_overbought_condition(price_data, rsi_period, overbought):
    """Condition for RSI overbought strategy."""
    rsi = technicals.calculate_rsi(price_data, rsi_period)
    return rsi[-1] > overbought

# --- Create Strategy Instances ---

def create_sma_crossover_strategy():
    return TradingStrategy(
        "SMA Crossover",
        [sma_crossover_condition],
        ["buy"],
        parameters={"short_sma_window": 10, "long_sma_window": 20}
    )

def create_rsi_overbought_strategy():
    return TradingStrategy(
        "RSI Overbought",
        [rsi_overbought_condition],
        ["sell"],
        parameters={"rsi_period": 14, "overbought": 70}
    )

# --- RSI Strategy (Updated) ---
class RSIStrategy(TradingStrategy):
    """A trading strategy based on the Relative Strength Index (RSI)."""

    def __init__(self, name="RSI Strategy", parameters={"rsi_period": 14, "overbought": 70, "oversold": 30}):
        # Use conditions and actions lists
        conditions = [
            lambda price_data, rsi_period, overbought: technicals.calculate_rsi(price_data, rsi_period)[-1] > overbought,
            lambda price_data, rsi_period, oversold: technicals.calculate_rsi(price_data, rsi_period)[-1] < oversold
        ]
        actions = ["sell", "buy"]
        super().__init__(name, conditions, actions, parameters)  # Correctly call superclass constructor