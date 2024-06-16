from abc import ABC, abstractmethod
import pickle
import time

import numpy as np

import os
import csv
import time

class TradeSignal:
    """Represents a trading signal."""
    def __init__(self, signal_type, source, predicted_entry_time, predicted_interval, trust_score):
        self.signal_type = signal_type
        self.source = source
        self.predicted_entry_time = predicted_entry_time
        self.predicted_interval = predicted_interval
        self.trust_score = trust_score
        self.entry_price = None
        self.exit_price = None
        self.entry_time = None
        self.exit_time = None
        self.result = None

    def is_valid(self, current_time):
        """Check if the signal is still valid based on time."""
        # Example: Signal is valid for 60 seconds
        return (current_time - self.timestamp) < 60

    def save_to_csv(self, file_path):
        """Saves the trade signal to a CSV file."""
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write header if file does not exist
                writer.writerow(['trade_id', 'source', 'signal_type', 'predicted_entry_time', 'predicted_interval', 'trust_score', 'entry_price', 'exit_price', 'entry_time', 'exit_time', 'result'])
            writer.writerow([self.trade_id, self.source, self.signal_type, self.predicted_entry_time, self.predicted_interval, self.trust_score, self.entry_price, self.exit_price, self.entry_time, self.exit_time, self.result])

    def update_csv(self, file_path):
        """Updates the trade signal in the CSV file."""
        temp_file_path = file_path + '.tmp'
        with open(file_path, mode='r', newline='') as file, open(temp_file_path, mode='w', newline='') as temp_file:
            reader = csv.reader(file)
            writer = csv.writer(temp_file)
            for row in reader:
                if row[0] == self.trade_id:
                    writer.writerow([self.trade_id, self.source, self.signal_type, self.predicted_entry_time, self.predicted_interval, self.trust_score, self.entry_price, self.exit_price, self.entry_time, self.exit_time, self.result])
                else:
                    writer.writerow(row)
        os.replace(temp_file_path, file_path)

    def __str__(self):
        return f"TradeSignal(source='{self.source}', type='{self.signal_type}', timestamp={self.timestamp}, predicted_entry_time={self.predicted_entry_time}, predicted_interval={self.predicted_interval}, trust_score={self.trust_score})"

class TradeStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def should_generate_signal(self, trading_model):
        """Determines if a trading signal should be generated.

        Args:
            trading_model (TradingModel): The trading model instance.

        Returns:
            bool: True if a signal should be generated, False otherwise.
        """
        pass

    @abstractmethod
    def generate_signal(self, trading_model, latest_price):
        """Generates a trading signal.

        Args:
            trading_model (TradingModel): The trading model instance.
            latest_price (float): The latest price of the asset.

        Returns:
            TradeSignal or None: The trading signal, or None if no signal is generated.
        """
        pass

    def save(self, filename):
        """Saves the strategy to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Loads a strategy from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)



class CombinedStrategy(TradeStrategy):
    def __init__(self, short_sma_window=10, long_sma_window=20, 
                sentiment_weight=0.5, rl_weight=0.5):
        super().__init__("Combined Strategy (SMA + Sentiment + RL)")
        self.short_sma_window = short_sma_window
        self.long_sma_window = long_sma_window
        self.sentiment_weight = sentiment_weight
        self.rl_weight = rl_weight

    def should_generate_signal(self, trading_model):
        latest_price = trading_model.data_manager.price_history[-1][0]
        price_history = [candle[0] for candle in trading_model.data_manager.price_history]

        # 1. SMA Crossover Check
        short_sma = self.calculate_sma(price_history, self.short_sma_window)
        long_sma = self.calculate_sma(price_history, self.long_sma_window)
        crossover_condition = short_sma[-1] > long_sma[-1] and short_sma[-2] <= long_sma[-2]

        # 2. Sentiment Check
        sentiment_score = trading_model.calculate_sentiment_score()
        sentiment_condition = sentiment_score > 0  # Adjust as needed

        # 3. RL Agent Check
        state = trading_model.rl_env._get_observation()
        action = trading_model.drlagent.act(state.reshape(1, -1))
        rl_condition = action == 1 if crossover_condition else action == 2  # Align with crossover

        # Combine conditions (adjust weights as needed)
        total_score = (crossover_condition + 
                    self.sentiment_weight * (1 if sentiment_condition else -1) + 
                    self.rl_weight * (1 if rl_condition else -1))

        return total_score > 0.5  # Adjust threshold as needed

    def generate_signal(self, trading_model, latest_price):
        return TradeSignal(self.name, "buy")  # Adjust buy/sell logic

    def calculate_sma(self, price_history, window):
        if len(price_history) < window:
            return [0] * len(price_history)
        return [sum(price_history[i-window:i]) / window for i in range(window, len(price_history))]

class RL_Strategy(TradeStrategy):
    def __init__(self):
        super().__init__("Reinforcement Learning Strategy")

    def should_generate_signal(self, trading_model):
        # Always generate a signal; let the RL agent decide the action
        return True

    def generate_signal(self, trading_model, latest_price):
        # Get the RL agent's action
        state = trading_model.rl_env._get_observation()  # Access through trading_model instance
        action = trading_model.drlagent.act(state.reshape(1, -1))
        signal_type = ["hold", "buy", "sell"][action]  # Map action to signal
        return TradeSignal(self.name, signal_type)


class SentimentStrategy(TradeStrategy):
    def __init__(self, sentiment_threshold=0.2):
        super().__init__("Sentiment-Based Strategy")
        self.sentiment_threshold = sentiment_threshold

    def should_generate_signal(self, trading_model):
        sentiment_score = trading_model.calculate_sentiment_score()  # Access through trading_model instance
        return abs(sentiment_score) >= self.sentiment_threshold  # Trade on strong sentiment

    def generate_signal(self, trading_model, latest_price):
        if trading_model.calculate_sentiment_score() > 0:  # Access through trading_model instance
            return TradeSignal(self.name, "buy")
        else:
            return TradeSignal(self.name, "sell")


class SMA_Crossover_Strategy(TradeStrategy):
    def __init__(self, short_sma_window=10, long_sma_window=20, sentiment_threshold=0.2):
        super().__init__("SMA Crossover with Sentiment")
        self.short_sma_window = short_sma_window
        self.long_sma_window = long_sma_window
        self.sentiment_threshold = sentiment_threshold

    def should_generate_signal(self, trading_model):
        # Get the latest price and price history from the trading model
        latest_price = trading_model.data_manager.price_history[-1][1]  # Use index 1 for price
        price_history = [candle[1] for candle in trading_model.data_manager.price_history]  # Use index 1 for price

        # Calculate SMAs
        short_sma = self.calculate_sma(price_history, self.short_sma_window)
        long_sma = self.calculate_sma(price_history, self.long_sma_window)

        # Check for SMA crossover AND if latest_price is above/below the SMAs (example logic)
        crossover_condition = short_sma[-1] > long_sma[-1] and short_sma[-2] <= long_sma[-2]
        price_above_smas = latest_price > short_sma[-1] and latest_price > long_sma[-1]  # Example condition

        # Check sentiment score
        sentiment_score = trading_model.calculate_sentiment_score()
        sentiment_condition = sentiment_score >= self.sentiment_threshold
        print(f"{self.name}: should_generate_signal() - crossover_condition: {crossover_condition}, price_above_smas: {price_above_smas}, sentiment_condition: {sentiment_condition}")
        # Generate a signal only if all conditions are met
        return crossover_condition and price_above_smas and sentiment_condition

    def generate_signal(self, trading_model, latest_price):
        print(f"{self.name}: generate_signal()")
        return TradeSignal(self.name, "buy")  # You can adjust buy/sell logic

    def calculate_sma(self, price_history, window):
        """Helper function to calculate SMA."""
        if len(price_history) < window:
            return [0] * len(price_history)
        return [sum(price_history[i-window:i]) / window for i in range(window, len(price_history))]


class RSI_Overbought_Strategy(TradeStrategy):
    def __init__(self, rsi_period=14, overbought_threshold=70, oversold_threshold=30):
        super().__init__("RSI Overbought/Oversold Strategy")
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def should_generate_signal(self, trading_model):
        price_history = [candle[0] for candle in trading_model.data_manager.price_history]
        rsi = self.calculate_rsi(price_history, self.rsi_period)

        # Generate signal if RSI is above overbought threshold or below oversold threshold
        return rsi[-1] > self.overbought_threshold or rsi[-1] < self.oversold_threshold

    def generate_signal(self, trading_model, latest_price):
        price_history = [candle[0] for candle in trading_model.data_manager.price_history]  # Access through trading_model instance
        rsi = self.calculate_rsi(price_history, self.rsi_period)

        if rsi[-1] > self.overbought_threshold:
            return TradeSignal(self.name, "sell")  # Sell when overbought
        elif rsi[-1] < self.oversold_threshold:
            return TradeSignal(self.name, "buy")

    def calculate_rsi(self, price_history, period):
        """Calculates the Relative Strength Index (RSI)."""
        if len(price_history) < period:
            return [50] * len(price_history)  # Return 50 as default for insufficient data

        deltas = np.diff(price_history)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period + 1e-8  # Add a small constant to prevent division by zero
        rs = up/down
        rsi = np.zeros_like(price_history)
        rsi[:period] = 100. - 100./(1.+rs)

        for i in range(period, len(price_history)):
            delta = deltas[i-1]  # Use i-1 because deltas is one element shorter
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period + 1e-8  # Add the constant here as well
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi


def create_sma_crossover_strategy(short_sma_window=10, long_sma_window=20, sentiment_threshold=0.2):
    """Creates an instance of the SMA Crossover with Sentiment strategy."""
    return SMA_Crossover_Strategy(short_sma_window, long_sma_window, sentiment_threshold)

def create_sentiment_strategy(sentiment_threshold=0.2):
    """Creates an instance of the Sentiment-Based Strategy."""
    return SentimentStrategy(sentiment_threshold)

def create_rl_strategy():
    """Creates an instance of the Reinforcement Learning Strategy."""
    return RL_Strategy()

def create_combined_strategy(short_sma_window=10, long_sma_window=20, sentiment_weight=0.5, rl_weight=0.5):
    """Creates an instance of the Combined Strategy."""
    return CombinedStrategy(short_sma_window, long_sma_window, sentiment_weight, rl_weight)

def create_rsi_overbought_strategy(rsi_period=14, overbought_threshold=70, oversold_threshold=30):
    """Creates an instance of the RSI Overbought/Oversold Strategy."""
    return RSI_Overbought_Strategy(rsi_period, overbought_threshold, oversold_threshold)