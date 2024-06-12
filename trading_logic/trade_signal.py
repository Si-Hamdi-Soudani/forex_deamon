import time
import os
import pickle

class TradeSignal:
    """Represents a trade signal with status tracking and details."""

    def __init__(self, entry_time, action, timeframe, strategy_name=None, signal_id=None):
        self.signal_id = signal_id or str(int(time.time() * 1000))  # Unique ID for the signal
        self.entry_time = entry_time
        self.action = action
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.status = "waiting"  # waiting, active, completed
        self.entry_price = None
        self.exit_time = None
        self.exit_price = None
        self.result = None  # win, loss
        self.candles = []

    def activate(self, entry_price):
        """Activates the trade signal when the entry time is reached."""
        self.status = "active"
        self.entry_price = entry_price

    def complete(self, exit_time, exit_price):
        """Completes the trade signal, determining the result."""
        self.status = "completed"
        self.exit_time = exit_time
        self.exit_price = exit_price

        if (self.action == "buy" and self.exit_price > self.entry_price) or (
            self.action == "sell" and self.exit_price < self.entry_price
        ):
            self.result = "win"
        else:
            self.result = "loss"

    def add_candle(self, timestamp, price):
        """Adds a candlestick data point to the signal."""
        self.candles.append((timestamp, price))

    def save(self, data_dir="data"):
        """Saves the trade signal to a file."""
        signal_file = os.path.join(data_dir, f"signal_{self.signal_id}.pkl")
        os.makedirs(data_dir, exist_ok=True)
        with open(signal_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(signal_id, data_dir="data"):
        """Loads a trade signal from a file."""
        signal_file = os.path.join(data_dir, f"signal_{signal_id}.pkl")
        try:
            with open(signal_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None