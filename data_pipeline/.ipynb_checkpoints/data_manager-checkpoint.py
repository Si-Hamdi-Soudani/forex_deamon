import collections
import time
import pickle
import os
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from trading_logic import trade_signal

class DataManager:
    """Manages data storage, checkpointing, and loading."""

    def __init__(self, price_history_length=120, checkpoint_interval=60):
        """Initializes the DataManager."""
        self.price_history_length = price_history_length
        self.checkpoint_interval = checkpoint_interval
        self.data_dir = "data"  # Directory to store data files
        self.price_history_file = os.path.join(self.data_dir, "price_history.pkl")
        self.checkpoint_file = os.path.join(self.data_dir, "checkpoint.pkl")
        self.signals_file = os.path.join(self.data_dir, "signals.pkl")
        
        # Load data on initialization
        self.price_history = self.load_data(self.price_history_file) or collections.deque(maxlen=self.price_history_length)
        self.last_checkpoint_time = self.load_checkpoint(self.checkpoint_file) or time.time()
        self.pending_signals = self.load_signals(self.data_dir) or []

    def save_data(self, data, filename):
        """Saves data to a pickle file."""
        os.makedirs(self.data_dir, exist_ok=True)  # Create data directory if it doesn't exist
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        """Loads data from a pickle file."""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_checkpoint(self, timestamp):
        """Saves a checkpoint with the last processed timestamp."""
        self.save_data(timestamp, self.checkpoint_file)

    def load_checkpoint(self, filename):
        """Loads the last processed timestamp from a checkpoint."""
        return self.load_data(filename)

    def save_signals(self, signals):
        """Saves pending trade signals to a file."""
        self.save_data(signals, self.signals_file)

    def load_signals(self, data_dir):  # Corrected argument: data_dir
        """Loads pending trade signals from files."""
        signals = []
        for filename in os.listdir(data_dir):
            if filename.startswith("signal_") and filename.endswith(".pkl"):
                signal_id = filename[7:-4]  # Extract signal ID from filename
                signal = trade_signal.TradeSignal.load(signal_id, data_dir)
                if signal:
                    signals.append(signal)
        return signals

    def update_price_history(self, timestamp, price):
        """Updates the price history and handles checkpointing."""
        #print(f"update_price_history called with timestamp: {timestamp}, price: {price}")

        # Resumption Logic:
        if timestamp <= self.last_checkpoint_time:
            print(f"Skipping duplicate timestamp: {timestamp}")
            return  # Skip this message

        self.price_history.append((timestamp, price))
        self.save_data(self.price_history, self.price_history_file)  # Save price history

        # Checkpointing
        if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
            self.save_checkpoint(timestamp)
            self.last_checkpoint_time = time.time()

    def perform_rfe(self, X, y, n_features):
        """Performs Recursive Feature Elimination (RFE) to select the most important features."""
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        return selector.support_

    def perform_pca(self, X, n_components):
        """Performs Principal Component Analysis (PCA) to reduce dimensionality."""
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return X_reduced