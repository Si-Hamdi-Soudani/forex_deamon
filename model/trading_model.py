import csv
import datetime
import itertools
import pickle
import random
import threading
import time
import json
from matplotlib import pyplot as plt
import numpy as np
from keras.src.models import Sequential, Model
from keras.src.utils import pad_sequences, to_categorical
from keras.src.layers import TextVectorization,LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Attention, Input, Lambda, Embedding, SpatialDropout1D
from sklearn.model_selection import ParameterGrid, train_test_split
import torch
from data_pipeline import websocket_client, data_manager
from trading_logic import trade_signal, trading_strategy
from utils import file_utils
import os
import logging
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, deque
import pandas as pd  # Import pandas for data loading
import tensorflow as tf
from sklearn.cluster import KMeans, DBSCAN  # Import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gym
from gym import spaces
import backtrader as bt
from keras.src.saving import load_model
from xgboost import XGBClassifier  # Import XGBoost

class DQNAgent:
    """A Deep Q-Network agent for reinforcement learning."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds the neural network model for the DQN."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Linear activation for Q-values
        model.compile(loss='mse', optimizer='adam')
        return model

    def replay(self, batch_size):
        """Trains the DQN using experiences from the replay memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action based on the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


class TradingEnvironment(gym.Env):
    """A custom trading environment for reinforcement learning."""

    def __init__(self, data_manager, window_size):
        super(TradingEnvironment, self).__init__()

        self.data_manager = data_manager
        self.window_size = window_size
        self.current_step = window_size  # Start at the end of the initial window
        self.current_price = None
        self.position = None  # None (no position), "long", or "short"
        self.balance = 10000  # Starting balance

        # Define action space: 0 - Hold, 1 - Buy, 2 - Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space (price history)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32)

    def reset(self):
        """Resets the environment to its initial state."""
        self.current_step = self.window_size
        self.current_price = self.data_manager.price_history[self.current_step]['close_price']  # Use 'close_price'
        self.position = None
        self.balance = 10000
        return self._get_observation()

    def step(self, action):
        """Takes a step in the environment based on the agent's action."""
        self.current_step += 1
        self.current_price = self.data_manager.price_history[self.current_step]['close_price']  # Use 'close_price'

        reward = 0
        if action == 1 and self.position is None:  # Buy
            self.position = "long"
        elif action == 2 and self.position is None:  # Sell
            self.position = "short"
        elif action == 0:  # Hold
            pass

        # Calculate reward based on position and price change
        if self.position == "long":
            reward = self.current_price - self.data_manager.price_history[self.current_step - 1]['close_price']  # Use 'close_price'
        elif self.position == "short":
            reward = self.data_manager.price_history[self.current_step - 1]['close_price'] - self.current_price  # Use 'close_price'

        done = self.current_step == len(self.data_manager.price_history) - 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Returns the current observation (price history)."""
        return np.array([candle['close_price'] for candle in self.data_manager.price_history[self.current_step - self.window_size:self.current_step]])  # Use 'close_price'
    

class BacktestStrategy(bt.Strategy):
    """A Backtrader strategy class for the beast."""

    params = (
        ('window_size', 10),
        ('n_clusters', 5),
        ('logistic_regression_C', 1.0),
        ('timeframe_multiplier', 1.0),
        ('position_size', None), # Add position_size parameter
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.preprocessed_data = []
        self.clustered_data = None
        self.prediction_models = {}
        self.kmeans = None
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit

        # Initialize KMeans model
        self.kmeans = KMeans(n_clusters=self.params.n_clusters, random_state=0)

        # Load completed candlesticks from the CSV file
        self.candlesticks = self.load_candlesticks('completed_candlesticks.csv')
        self.current_candle_index = self.params.window_size # Start at the end of the initial window

    def next(self):
        """Executes trading logic for each bar."""

        # Check if we have enough candlestick data
        if self.current_candle_index >= len(self.candlesticks):
            return 

        current_candle = self.candlesticks[self.current_candle_index]
        close = current_candle['close_price']

        # Preprocess the data
        preprocessed_data = self.data_manager.preprocess_candlestick_data(
            self.candlesticks, window_size=self.params.window_size
        )

        # Cluster the data
        clustered_data = self.identify_candlestick_patterns(preprocessed_data)

        # Train the prediction model
        self.train_prediction_model(clustered_data)

        # Get the cluster ID for the current bar
        cluster_id = self.kmeans.predict(
            np.array([close, close, close, close]).reshape(1, -1)
        )[0]

        # Determine the timeframe
        timeframe = self.determine_timeframe(cluster_id, np.array([close, close, close, close]))

        # Make a prediction using the trained model
        prediction = self.predict_price_movement(cluster_id, np.array([close, close, close, close]))

        # Generate a trade signal
        if prediction == 1:  # Buy
            self.buy(size=self.params.position_size)  # Buy with the calculated position size
            self.stop_loss = self.buyprice * (1 - self.stop_loss_percentage)
            self.take_profit = self.buyprice * (1 + self.take_profit_percentage)
        elif prediction == 0:  # Sell
            self.sell(size=self.params.position_size)  # Sell with the calculated position size
            self.stop_loss = self.sellprice * (1 + self.stop_loss_percentage)
            self.take_profit = self.sellprice * (1 - self.take_profit_percentage)

        # Check for stop-loss or take-profit
        if self.position:
            if self.dataclose[0] <= self.stop_loss:
                self.close()
                print(f"Stop-loss triggered at {self.dataclose[0]}")
            elif self.dataclose[0] >= self.take_profit:
                self.close()
                print(f"Take-profit triggered at {self.dataclose[0]}")

        # Move to the next candle
        self.current_candle_index += 1

    def load_candlesticks(self, filename='completed_candlesticks.csv'):
        """Loads candlestick data from a CSV file."""
        return self.data_manager.load_candlesticks(filename)



class TradingModel:
    """The core trading model (the beast's brain)."""

    def __init__(self, data_manager, short_sma_window=10, long_sma_window=20, cooldown_period=60):
        """Initializes the trading model."""
        self.params = {
            'window_size': 10,
            'n_clusters': 5,
            'logistic_regression_C': 1.0,
            'timeframe_multiplier': 1.0
        }
        self.last_strategy_generation_time = 0
        self.signal_generation_limit = 5  # Limit the number of signals per hour
        self.signals_generated = 0
        self.data_manager = data_manager
        self.short_sma_window = short_sma_window
        self.long_sma_window = long_sma_window
        self.current_position = None  # None (no position), "buy", or "sell"
        self.trade_history = []  # Stores a history of trades
        self.websocket_client = None
        self.model_file = "trading_model.pkl"
        self.active_trades = []  # Stores currently active TradeSignal objects
        self.pending_signals = self.data_manager.pending_signals  # Load pending signals from DataManager
        self.cooldown_period = cooldown_period  # Cooldown period in seconds
        self.last_trade_time = 0
        self.sentiment_history = defaultdict(list)
        logging.basicConfig(filename='beast.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # **FIX: Use correct strategy creation functions and load strategies**
        self.strategies = [
            trading_strategy.create_sma_crossover_strategy(),
            trading_strategy.create_rsi_overbought_strategy()
        ]
        self.load_strategies() # Load from files, appending to existing

        # Define the hybrid CNN-LSTM model with attention
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None, 1)),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            tf.keras.layers.Lambda(lambda x: [x, x]),  # Duplicate the LSTM output for query and value
            Attention(),
            LSTM(50),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")

        # Get input shape from data manager
        input_shape = (None, self.data_manager.window_size, self.data_manager.features)

        # Build CNN-LSTM-Attention model
        try:
            self.cnn_lstm_attention_model = file_utils.load_object('trading_model.pkl')
            print("Loaded existing model.")
        except Exception as e:
            print("Model not found. Training a new model...", e)
            self.cnn_lstm_attention_model = self.build_cnn_lstm_attention_model(input_shape)
            self.train_model()  # Train the model
            file_utils.save_object(self.cnn_lstm_attention_model, 'trading_model.pkl')  # Save the trained model
            print("Model trained and saved.")
        
        # Load model state on initialization
        self.load_model_state()

        # Initialize ensemble models
        self.ensemble_models = []

        logistic_model = LogisticRegression()
        random_forest_model = RandomForestClassifier()

        self.add_ensemble_model(logistic_model, "Logistic Regression")
        self.add_ensemble_model(random_forest_model, "Random Forest")

        self.sentiment_index = None
        self.sufficient_data = False  # Flag to track if enough data is available
        self.minimum_candlesticks = 100  # Minimum number of candlesticks required

        # Initialize parameters for the data-driven strategy
        self.params = {
            'window_size': 10,
            'n_clusters': 5,
            'logistic_regression_C': 1.0,
            'timeframe_multiplier': 1.0
        }

        # Initialize KMeans model
        self.kmeans = KMeans(n_clusters=self.params['n_clusters'], random_state=0)

        # Initialize Logistic Regression model
        self.logistic_regression = LogisticRegression(C=self.params['logistic_regression_C'])

        # Initialize timeframe multiplier
        self.timeframe_multiplier = self.params['timeframe_multiplier']

        # Initialize the preprocessed data and clustered data
        self.preprocessed_data = []
        self.clustered_data = None
        self.prediction_models = {}
        self.last_candle_start_time = None
        self.candlestick_data = deque(maxlen=1)  # To store current candle data

        # Initialize the RL trading environment
        self.rl_candlesticks = self.data_manager.load_candlesticks('completed_candlesticks.csv')
        self.drlagent = DQNAgent(state_size=self.params['window_size'], action_size=3)
        self.rl_env = TradingEnvironment(self.data_manager, self.params['window_size'])

        # Initialize lists to store predictions and actual outcomes
        self.predictions = []
        self.actual_outcomes = []
        self.accuracy = 0.0
        self.completed_trades = []
        self.completed_candlesticks_file = "completed_candlesticks.csv"

        # Initialize BERT/FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        self.position_size_percentage = 0.1

        self.sentiment_model_path = 'sentiment_model.h5'
        self.tokenizer_path = 'tokenizer.pickle'
        self.max_vocab = 5000
        self.max_len = 100
        self.load_or_train_sentiment_model()

    def start_websocket_feed(self):
        """Starts the websocket client in a separate thread."""
        self.websocket_client = websocket_client.CoinMarketCapWebsocketClient(
            on_message_callback=self.on_price_update,
            on_error_callback=self.on_error,
            on_close_callback=self.on_close,
        )
        websocket_thread = threading.Thread(target=self.websocket_client.connect)
        websocket_thread.start()
        self.websocket_connected = True

    def stop_websocket_feed(self):
        """Stops the websocket client."""
        if self.websocket_client:
            self.websocket_client.disconnect()
            self.websocket_connected = False

    def on_price_update(self, completed_candle):
        """Handles incoming price updates from the websocket."""
        if completed_candle:
            self.data_manager.price_history.append(completed_candle) 
            self.evaluate_and_execute_trade()

    def on_error(self, error):
        """Handles errors from the websocket client."""
        logging.error(f"Websocket error: {error}")

    def on_close(self, close_status_code, close_msg):
        """Handles closure of the websocket connection."""
        logging.info(f"Websocket connection closed with code: {close_status_code}, message: {close_msg}")
        # Restart the websocket connection automatically
        time.sleep(5)
        self.start_websocket_feed()

    def load_model_state(self):
        """Loads the trading model's state from a file."""
        try:
            data = pickle.load(open(self.model_file, 'rb'))
            self.current_position = data['current_position']
            self.trade_history = data['trade_history']
            self.last_trade_time = data['last_trade_time']
            print("Model state loaded successfully.")
        except FileNotFoundError:
            print("Model state file not found. Starting fresh.")
        except Exception as e:
            print(f"Error loading model state: {e}")

    def save_model_state(self):
        """Saves the trading model's state to a file."""
        data = {
            'current_position': self.current_position,
            'trade_history': self.trade_history,
            'last_trade_time': self.last_trade_time
        }
        pickle.dump(data, open(self.model_file, 'wb'))
        print("Model state saved successfully.")

    def add_ensemble_model(self, model, name):
        """Adds a model to the ensemble."""
        self.ensemble_models.append({
            'model': model,
            'name': name,
            'trained': False  # Initially, the model is not trained
        })

    def train_ensemble_models(self, X, y):
        """Trains the ensemble models."""
        for model_data in self.ensemble_models:
            model = model_data['model']
            if isinstance(model, (LogisticRegression, RandomForestClassifier)):
                model.fit(X, y)
                model_data['trained'] = True
                print(f"{model_data['name']} model trained.")

    def predict_with_ensemble(self, X):
        """Makes predictions using the ensemble models."""
        if not all(model_data['trained'] for model_data in self.ensemble_models):
            print("Not all ensemble models are trained. Skipping prediction.")
            return None

        predictions = []
        for model_data in self.ensemble_models:
            model = model_data['model']
            if isinstance(model, (LogisticRegression, RandomForestClassifier)):
                prediction = model.predict_proba(X)[:, 1]  # Get probability for class 1
                predictions.append(prediction)

        if predictions:
            return np.mean(predictions, axis=0)  # Average the probabilities
        else:
            return None

    def evaluate_and_execute_trade(self):
        """Evaluates trading signals and executes trades if conditions are met."""
        # Check if enough time has passed since the last trade
        if time.time() - self.last_trade_time < self.cooldown_period:
            return

        # Check if enough data is available for trading
        if not self.sufficient_data:
            if len(self.data_manager.price_history) >= self.minimum_candlesticks:
                self.sufficient_data = True
                print("Sufficient data available. Ready to trade.")
            else:
                print(f"Waiting for more data... ({len(self.data_manager.price_history)}/{self.minimum_candlesticks} candlesticks)")
                return

        # Check for pending signals first
        if self.pending_signals:
            self.process_pending_signals()
            return  # Exit to avoid generating new signals while processing pending ones

        # Generate a new trading signal
        self.generate_trading_signal()

    def process_pending_signals(self):
        """Processes pending trading signals."""
        for signal in self.pending_signals:
            if signal.is_valid(self.data_manager.price_history[-1]['close_price']):  # Check if the signal is still valid
                self.execute_trade(signal)
                self.pending_signals.remove(signal)
                self.data_manager.save_signals(self.pending_signals)  # Save updated pending signals
                return  # Process only one signal at a time

    def generate_trading_signal(self):
        """Generates a trading signal based on the selected strategy."""
        # Check if enough time has passed since the last strategy generation
        if time.time() - self.last_strategy_generation_time < 3600:  # 1 hour cooldown
            return

        # Reset the signal counter
        self.signals_generated = 0

        # Get the latest price from the data manager
        latest_price = self.data_manager.price_history[-1]['close_price']

        for strategy in self.strategies:
            if strategy.should_generate_signal(latest_price, self.data_manager.price_history):
                signal = strategy.generate_signal(latest_price)
                if signal:
                    # Check if the signal generation limit is reached
                    if self.signals_generated < self.signal_generation_limit:
                        # Check if the signal is valid
                        if signal.is_valid(latest_price):
                            self.execute_trade(signal)
                            self.signals_generated += 1
                        else:
                            print(f"Signal from {strategy.name} is not valid. Skipping...")
                    else:
                        print("Signal generation limit reached for this hour.")
                        break  # Stop generating signals for this hour

        # Update the last strategy generation time
        self.last_strategy_generation_time = time.time()

    def execute_trade(self, signal):
        """Executes a trade based on the given trading signal."""
        # Check if a trade is already in progress
        if self.current_position is not None:
            print(f"Trade already in progress: {self.current_position}")
            return

        # Check if enough time has passed since the last trade
        if time.time() - self.last_trade_time < self.cooldown_period:
            print(f"Cooldown period in effect. Waiting {self.cooldown_period} seconds before next trade.")
            self.pending_signals.append(signal)  # Add the signal to pending signals
            self.data_manager.save_signals(self.pending_signals)  # Save pending signals
            return

        latest_price = self.data_manager.price_history[-1]['close_price']

        if signal.signal_type == "buy":
            self.current_position = "buy"
            self.buy_order(latest_price, signal)
        elif signal.signal_type == "sell":
            self.current_position = "sell"
            self.sell_order(latest_price, signal)

        self.last_trade_time = time.time()
        self.save_model_state()  # Save the model state after executing the trade

    def buy_order(self, latest_price, signal):
        """Executes a buy order."""
        self.trade_history.append({
            'timestamp': time.time(),
            'price': latest_price,
            'type': 'buy',
            'signal_source': signal.source
        })
        print(f"Buy order executed at {latest_price}, signal source: {signal.source}")

    def sell_order(self, latest_price, signal):
        """Executes a sell order."""
        self.trade_history.append({
            'timestamp': time.time(),
            'price': latest_price,
            'type': 'sell',
            'signal_source': signal.source
        })
        print(f"Sell order executed at {latest_price}, signal source: {signal.source}")

    def close_open_position(self):
        """Closes any open position."""
        if self.current_position is not None:
            latest_price = self.data_manager.price_history[-1]['close_price']
            if self.current_position == "buy":
                self.sell_order(latest_price, trade_signal.TradeSignal("close_position", "sell"))
            elif self.current_position == "sell":
                self.buy_order(latest_price, trade_signal.TradeSignal("close_position", "buy"))
            self.current_position = None
            self.save_model_state()

    def load_strategies(self):
        """Loads trading strategies from files if available."""
        for filename in os.listdir("."):
            if filename.startswith("strategy_") and filename.endswith(".pkl"):
                try:
                    strategy = trading_strategy.TradeStrategy.load(filename[:-4])  # Remove ".pkl"
                    if strategy:
                        self.strategies.append(strategy)
                        print(f"Loaded strategy from {filename}")
                except Exception as e:
                    print(f"Error loading strategy from {filename}: {e}")

    def save_strategies(self):
        """Saves current trading strategies to files."""
        for i, strategy in enumerate(self.strategies):
            try:
                filename = f"strategy_{strategy.name}_{i}.pkl"
                strategy.save(filename)
                print(f"Saved strategy to {filename}")
            except Exception as e:
                print(f"Error saving strategy {strategy.name}: {e}")

    def build_cnn_lstm_attention_model(self, input_shape):
        """Builds the CNN-LSTM-Attention model architecture."""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Attention(),
            Flatten(),
            Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        """Trains the CNN-LSTM-Attention model."""
        # Load preprocessed candlestick data
        candlesticks = self.data_manager.load_candlesticks(self.completed_candlesticks_file)
        if candlesticks:
            # Preprocess the candlestick data
            preprocessed_data = self.data_manager.preprocess_candlestick_data(candlesticks)
            X, y = zip(*preprocessed_data)
            X = np.array(X)
            y = np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for CNN

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            self.cnn_lstm_attention_model.fit(X_train, y_train, epochs=10, batch_size=32)

            # Evaluate the model
            _, accuracy = self.cnn_lstm_attention_model.evaluate(X_test, y_test)
            print("Model Accuracy:", accuracy)
        else:
            print("No candlestick data found for training.")

    def predict_price_movement(self, data):
        """Predicts the price movement using the trained model."""
        if self.cnn_lstm_attention_model is not None:
            data = np.array(data)
            data = data.reshape((1, data.shape[0], 1))  # Reshape for CNN
            prediction = self.cnn_lstm_attention_model.predict(data)
            return 1 if prediction > 0.5 else 0  # 1 for up, 0 for down
        else:
            print("Model not trained. Cannot predict.")
            return None

    def optimize_hyperparameters(self, candlesticks):
        """Optimizes the hyperparameters of the trading strategy."""

        def objective(params):
            """Objective function for hyperparameter optimization."""
            # Create a new BacktestStrategy instance with the given parameters
            strategy = BacktestStrategy(self.data_manager, **params)

            # Run the backtest
            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy)
            cerebro.adddata(bt.feeds.PandasData(dataname=pd.DataFrame(candlesticks)))
            cerebro.run()

            # Return the negative of the final portfolio value (to minimize)
            return -cerebro.broker.getvalue()

        # Define the hyperparameter space
        space = {
            'window_size': hp.choice('window_size', range(5, 21)),
            'n_clusters': hp.choice('n_clusters', range(2, 11)),
            'logistic_regression_C': hp.uniform('logistic_regression_C', 0.01, 10.0),
            'timeframe_multiplier': hp.uniform('timeframe_multiplier', 0.5, 2.0)
        }

        # Run the hyperparameter optimization using Tree-structured Parzen Estimator (TPE)
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,  # Number of optimization iterations
                    trials=trials)

        # Print the best hyperparameters
        print("Best Hyperparameters:", best)

        # Update the strategy's parameters with the best ones
        self.params.update(best)

    def identify_candlestick_patterns(self, preprocessed_data):
        """Identifies candlestick patterns and clusters them using KMeans."""
        # Extract features from the preprocessed data
        features = np.array([data_point[0] for data_point in preprocessed_data])

        # Fit KMeans to the features
        self.kmeans.fit(features)

        # Get cluster labels for each data point
        cluster_labels = self.kmeans.labels_

        # Create a list to store the clustered data
        clustered_data = []

        # Group data points by cluster label
        for i in range(self.params['n_clusters']):
            cluster_data = [data_point for data_point, label in zip(preprocessed_data, cluster_labels) if label == i]
            clustered_data.append(cluster_data)

        return clustered_data

    def train_prediction_model(self, clustered_data):
        """Trains a prediction model (Logistic Regression) for each cluster."""
        for cluster_id, cluster_data in enumerate(clustered_data):
            # Check if the cluster has enough data
            if len(cluster_data) > self.params['window_size']:
                # Extract features and target variable
                X = np.array([data_point[0] for data_point in cluster_data])
                y = np.array([data_point[1] for data_point in cluster_data])

                # Train the Logistic Regression model
                self.logistic_regression.fit(X, y)

                # Store the trained model
                self.prediction_models[cluster_id] = self.logistic_regression

    def determine_timeframe(self, cluster_id, data_point):
        """Determines the timeframe for trading based on cluster characteristics."""
        # Calculate the average distance of data points from the cluster center
        cluster_center = self.kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(np.array(data_point) - cluster_center, axis=0)
        average_distance = np.mean(distances)

        # Adjust the timeframe multiplier based on the average distance
        timeframe_multiplier = self.params['timeframe_multiplier'] * (1 + average_distance)

        # Return the calculated timeframe
        return int(timeframe_multiplier)

    def update_sentiment_history(self, sentiment):
        """Updates the sentiment history with the latest sentiment value."""
        self.sentiment_history['sentiment'].append(sentiment)
        self.sentiment_history['timestamp'].append(time.time())

        # Limit the history length
        max_history_length = 1000  # Keep track of the last 1000 sentiment values
        self.sentiment_history['sentiment'] = self.sentiment_history['sentiment'][-max_history_length:]
        self.sentiment_history['timestamp'] = self.sentiment_history['timestamp'][-max_history_length:]

    def calculate_sentiment_score(self):
        """Calculates the sentiment score based on the sentiment history."""
        if len(self.sentiment_history['sentiment']) == 0:
            return 0  # Return 0 if there's no sentiment history yet

        # Calculate a weighted average sentiment score, giving more weight to recent sentiments
        weights = np.exp(np.linspace(0, -1, len(self.sentiment_history['sentiment'])))
        weighted_sentiment = np.average(self.sentiment_history['sentiment'], weights=weights)
        return weighted_sentiment

    def load_or_train_sentiment_model(self):
        """Loads a pre-trained sentiment analysis model or trains a new one."""
        try:
            # Try to load the tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)

            # Try to load the model
            self.model = load_model(self.sentiment_model_path)
            print("Sentiment analysis model loaded successfully.")
        except (FileNotFoundError, OSError):
            print("Sentiment analysis model not found. Training a new model...")
            self.train_sentiment_model()

    def train_sentiment_model(self):
        """Trains a new sentiment analysis model."""
        # 1. Prepare your data
        texts, labels = self.data_manager.prepare_sentiment_analysis_data()

        # 2. Tokenize the text
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_vocab)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)

        # 3. Create the model
        model = Sequential([
            Embedding(self.max_vocab, 128, input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])

        # 4. Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # 5. Train the model
        model.fit(padded_sequences, np.array(labels), epochs=10)

        # 6. Save the model and tokenizer
        model.save(self.sentiment_model_path)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)

        self.model = model
        self.tokenizer = tokenizer
        print("Sentiment analysis model trained and saved.")

    def analyze_sentiment(self, text):
        """Analyzes the sentiment of a given text using the trained model."""
        if self.model is None:
            print("Sentiment analysis model not loaded. Cannot analyze sentiment.")
            return 0

        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        prediction = self.model.predict(padded_sequence)[0][0]
        return prediction

    def execute_rl_trade(self, action):
        """Executes a trade based on the RL agent's action."""
        current_price = self.data_manager.price_history[-1]['close_price']

        if action == 1:  # Buy
            if self.current_position is None:
                self.buy_order(current_price, trade_signal.TradeSignal("RL Agent", "buy"))
                self.current_position = "buy"
        elif action == 2:  # Sell
            if self.current_position is None:
                self.sell_order(current_price, trade_signal.TradeSignal("RL Agent", "sell"))
                self.current_position = "sell"
        else:  # Hold
            pass

    def train_rl_agent(self, episodes=1000):
        """Trains the RL agent using the trading environment."""
        for episode in range(episodes):
            state = self.rl_env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.drlagent.act(state)
                next_state, reward, done, _ = self.rl_env.step(action)
                self.drlagent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

            # Replay experience for training
            if len(self.drlagent.memory) > 32:
                self.drlagent.replay(32)

    def backtest_strategy(self, candlesticks):
        """Backtests the trading strategy using historical data."""
        # Create a cerebro instance
        cerebro = bt.Cerebro()

        # Add data feed
        data = bt.feeds.PandasData(dataname=pd.DataFrame(candlesticks))
        cerebro.adddata(data)

        # Add the strategy
        cerebro = bt.Cerebro()
        cerebro.addstrategy(BacktestStrategy)
        cerebro.adddata(bt.feeds.PandasData(dataname=pd.DataFrame(candlesticks)))

        # Set up the backtest parameters
        cerebro.broker.setcash(100000)  # Initial capital
        cerebro.broker.setcommission(commission=0.001)  # Trading commission

        # Run the backtest
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Plot the results (optional)
        cerebro.plot()

    def run_genetic_algorithm(self, candlesticks, population_size=50, generations=100):
        """Runs a genetic algorithm to optimize trading strategy parameters."""
        # Define the genetic algorithm parameters
        creator.create("FitnessMax", base.Fitness, values=(1.0,))  # Maximize a single fitness value
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Define the gene space (parameters to optimize)
        gene_space = [
            {'name': 'short_sma_window', 'type': 'int', 'min': 5, 'max': 20},
            {'name': 'long_sma_window', 'type': 'int', 'min': 20, 'max': 50},
        ]

        # Define the evaluation function
        def evaluate_strategy(individual):
            # Create a strategy with the individual's genes
            params = {gene['name']: individual[i] for i, gene in enumerate(gene_space)}
            strategy = BacktestStrategy(self.data_manager, **params)

            # Run the backtest
            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy)
            cerebro.adddata(bt.feeds.PandasData(dataname=pd.DataFrame(candlesticks)))
            cerebro.run()

            # Return the final portfolio value as fitness
            return cerebro.broker.getvalue(),

        # Initialize the toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_int(gene['min'], gene['max']) for gene in gene_space), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_strategy)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[gene['min'] for gene in gene_space],
                         high=[gene['max'] for gene in gene_space], indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create the initial population
        pop = toolbox.population(n=population_size)

        # Run the genetic algorithm
        hof = tools.HallOfFame(1)  # Keep track of the best individual
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, num_gen=generations,
                                           stats=stats, halloffame=hof, verbose=True)

        # Get the best individual
        best_individual = hof[0]
        best_params = {gene['name']: best_individual[i] for i, gene in enumerate(gene_space)}

        # Print the best parameters and fitness
        print("Best Parameters:", best_params)
        print("Best Fitness:", hof[0].fitness.values[0])

        return best_params