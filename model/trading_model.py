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
from keras.src.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Attention, Input, Lambda
from sklearn.model_selection import ParameterGrid
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
from collections import defaultdict, deque
import pandas as pd  # Import pandas for data loading
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from backtrader.analyzers import SharpeRatio, DrawDown, AnnualReturn
from backtrader.cerebro import Cerebro
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gym
from gym import spaces
import backtrader as bt

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
        self.current_price = self.data_manager.price_history[self.current_step][1]
        self.position = None
        self.balance = 10000
        return self._get_observation()

    def step(self, action):
        """Takes a step in the environment based on the agent's action."""
        self.current_step += 1
        self.current_price = self.data_manager.price_history[self.current_step][1]

        reward = 0
        if action == 1 and self.position is None:  # Buy
            self.position = "long"
        elif action == 2 and self.position is None:  # Sell
            self.position = "short"
        elif action == 0:  # Hold
            pass

        # Calculate reward based on position and price change
        if self.position == "long":
            reward = self.current_price - self.data_manager.price_history[self.current_step - 1][1]
        elif self.position == "short":
            reward = self.data_manager.price_history[self.current_step - 1][1] - self.current_price

        done = self.current_step == len(self.data_manager.price_history) - 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Returns the current observation (price history)."""
        return np.array([price for _, price in self.data_manager.price_history[self.current_step - self.window_size:self.current_step]])
    

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

        # Initialize KMeans model
        self.kmeans = KMeans(n_clusters=self.params.n_clusters, random_state=0)

    def next(self):
        """Executes trading logic for each bar."""
        # Get the current bar's data
        current_bar = len(self) - 1
        close = self.dataclose[0]

        # Preprocess the data
        preprocessed_data = self.data_manager.preprocess_candlestick_data(
            self.data_manager.price_history, window_size=self.params.window_size
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
        self.cnn_lstm_attention_model = self.build_cnn_lstm_attention_model(input_shape)


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
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")  # Load FinBERT tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")  # Load FinBERT model

        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        self.position_size_percentage = 0.1

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

    def on_price_update(self, price_data):
        """Handles incoming price updates from the websocket."""
        try:
            data = json.loads(price_data)
            if data.get("d") and data["d"]["id"] == 1:
                price = data["d"]["p"]
                timestamp = int(data["t"]) / 1000
                self.data_manager.update_price_history(timestamp, price)
                self._update_candlestick_data(timestamp, price)
                self.evaluate_and_execute_trade()
        except Exception as e:
            print(f"Error processing price update from function on_price_update from trading_model: {e}")

    def on_error(self, error):
        """Handles errors from the websocket client."""
        logging.error(f"Websocket error: {error}")

    def on_close(self, close_status_code, close_msg):
        """Handles closure of the websocket connection."""
        logging.info(f"Websocket connection closed with code: {close_status_code}, message: {close_msg}")
        # Restart the websocket connection automatically
        time.sleep(5)
        self.start_websocket_feed()
    def save_model_state(self):
        """Saves the trading model's state to a file."""
        model_state = {
            "strategies": self.strategies,
            "trade_history": self.trade_history,
            "active_trades": self.active_trades,
            "pending_signals": self.pending_signals,
            "last_trade_time": self.last_trade_time,
            "sentiment_history": self.sentiment_history,
            "ensemble_models": self.ensemble_models,
            "sentiment_index": self.sentiment_index,
            "sufficient_data": self.sufficient_data,
            "predictions": self.predictions,
            "actual_outcomes": self.actual_outcomes,
            "accuracy": self.accuracy,
        }
        file_utils.save_object(model_state, self.model_file)

    def load_model_state(self):
        """Loads the trading model's state from a file."""
        try:
            # Attempt to load the model state
            model_state = file_utils.load_object('trading_model.pkl')
            
            # Check if the model state is None (file not found or empty)
            if model_state is None:
                print("Model state file 'trading_model.pkl' not found or is empty. Starting with default state.")
                # Initialize default states or perform any necessary setup
                self.initialize_default_states()
            else:
                # Load states from model_state
                self.strategies = model_state.get("strategies", [])
                self.trade_history = model_state.get("trade_history", [])
                self.active_trades = model_state.get("active_trades", [])
                self.pending_signals = model_state.get("pending_signals", [])
                self.last_trade_time = model_state.get("last_trade_time", 0)
                self.predictions = model_state.get("predictions", [])
                self.actual_outcomes = model_state.get("actual_outcomes", [])
                self.accuracy = model_state.get("accuracy", 0.0)
                # Add more fields as necessary
        except FileNotFoundError:
            print("Model state file 'trading_model.pkl' not found. Starting with default state.")
            # Initialize default states or perform any necessary setup
            self.initialize_default_states()
        except Exception as e:
            print(f"Error loading model state: {e}")
            # Handle other exceptions or initialize default states
            self.initialize_default_states()

    def initialize_default_states(self):
        """Initializes default states for the model."""
        self.strategies = []  # Initialize with default strategies if any
        self.trade_history = []
        self.active_trades = []
        self.pending_signals = []
        self.last_trade_time = 0
        self.predictions = []
        self.actual_outcomes = []
        self.accuracy = 0.0

    def save_strategies(self):
        """Saves the trading strategies to files."""
        for strategy in self.strategies:
            strategy.save()

    def load_strategies(self):
        """Loads trading strategies from files."""
        for filename in os.listdir("data"):
            if filename.startswith("strategy_"):
                strategy = trading_strategy.TradingStrategy.load(filename.split("_")[1].split(".")[0])
                self.strategies.append(strategy)

    def add_ensemble_model(self, model, model_name):
        """Adds a model to the ensemble."""
        self.ensemble_models.append((model, model_name))

    def predict_ensemble(self, price_data):
        """Makes predictions using the ensemble of models."""
        predictions = []
        for model, model_name in self.ensemble_models:
            print(f"Predicting using {model_name}")
            # Train the model and make predictions
            if model_name == "Logistic Regression":
                model.fit(np.array(price_data).reshape(-1, 1), np.array([0] * len(price_data)))
                prediction = model.predict(np.array(price_data).reshape(-1, 1))
            elif model_name == "Random Forest":
                model.fit(np.array(price_data).reshape(-1, 1), np.array([0] * len(price_data)))
                prediction = model.predict(np.array(price_data).reshape(-1, 1))
            predictions.append(prediction)
        # Combine predictions (e.g., majority voting)
        return predictions[0]
    
    def predict_price_movement(self, features, cluster_id):
        """Predicts the price movement (up or down) based on the identified pattern."""
        model = self.prediction_models[cluster_id]
        prediction = model.predict(features)
        return prediction[0]

    def evaluate_and_execute_trade(self):
        """Evaluates trading strategies and executes trades based on signals."""
        current_time = time.time()

        # Check if cooldown period is active
        if current_time - self.last_trade_time < self.cooldown_period:
            remaining_time = self.cooldown_period - (current_time - self.last_trade_time)
            print(f"Cooldown period active. {remaining_time:.2f} seconds remaining.")
            return

        # Reset signals generated count every hour
        if current_time - self.last_strategy_generation_time >= 3600:
            self.signals_generated = 0
            self.last_strategy_generation_time = current_time

        # Limit trade frequency
        max_trades_per_minute = 1
        trades_this_minute = 0
        last_minute = int(current_time // 60)

        # Fetch and analyze news
        self.data_manager.fetch_coindesk_news()
        self.data_manager.fetch_binance_news()
        self.data_manager.cluster_news_articles()  # Cluster articles
        news_articles = self.data_manager.get_news_articles()

        # Calculate sentiment index
        self.sentiment_index = self._calculate_sentiment_index(news_articles)
        print(f"Overall Sentiment Index: {self.sentiment_index}")

        # Analyze sentiment trends
        self._analyze_sentiment_trends(news_articles)

        # Evaluate and execute trades based on live data
        if len(self.data_manager.price_history) >= self.minimum_candlesticks:
            preprocessed_data = self.data_manager.preprocess_candlestick_data(self.data_manager.price_history, window_size=10)  # Adjust window_size as needed
            clustered_data = self.identify_candlestick_patterns(preprocessed_data)
            self.train_prediction_model(clustered_data)
            self.clustered_data = clustered_data  

            # Check if cooldown period is active
            if current_time - self.last_trade_time < self.cooldown_period:
                remaining_time = self.cooldown_period - (current_time - self.last_trade_time)
                print(f"Cooldown period active. {remaining_time:.2f} seconds remaining.")
                return

            # Limit trade frequency
            current_minute = int(current_time // 60)
            if current_minute == last_minute and trades_this_minute >= max_trades_per_minute:
                print(f"Reached maximum trades ({max_trades_per_minute}) for this minute. Skipping signal.")
                return

            # Check signal generation limit
            if self.signals_generated >= self.signal_generation_limit:
                print(f"Signal generation limit ({self.signal_generation_limit}) reached. Skipping signal.")
                return

            # Train the CNN-LSTM-Attention model if enough data is available
            self.train_cnn_lstm_attention_model()

            # Use the DRL agent for trading decisions
            if self.rl_env:
                state = self.rl_env.reset()
                state = state.reshape(1, self.params['window_size'])  # Reshape for the DQN

                # Train the DRL agent
                for _ in range(100):  # Adjust the number of training steps as needed
                    action = self.drlagent.act(state)
                    next_state, reward, done, _ = self.rl_env.step(action)
                    next_state = next_state.reshape(1, self.params['window_size'])
                    self.drlagent.remember(state, action, reward, next_state, done)
                    state = next_state

                    if done:
                        state = self.rl_env.reset()
                        state = state.reshape(1, self.params['window_size'])

                    self.drlagent.replay(batch_size=32)  # Adjust batch size as needed

                # Get the DRL agent's prediction
                action = self.drlagent.act(state)

                if action == 1:  # Buy
                    action = "buy"
                elif action == 2:  # Sell
                    action = "sell"
                else:
                    action = None

            # Make a prediction using the trained model
            if len(self.candlestick_data) >= self.params.window_size:
                last_candle = self.candlestick_data[-1]
                features = np.array([
                    last_candle[1], last_candle[2], last_candle[3], last_candle[4]
                ])
                features = (features - np.mean(features)) / np.std(features)
                features = features.reshape(1, self.params['window_size'], 1)  # Reshape for CNN-LSTM

                # Predict price movement using the CNN-LSTM-Attention model
                prediction = self.cnn_lstm_attention_model.predict(features)[0][0]  # Get the prediction value

                # Identify the cluster
                cluster_id = self.kmeans.predict(features.reshape(1, -1))[0]  # Get the cluster ID from KMeans

                # Determine the timeframe
                timeframe = self.determine_timeframe(cluster_id, features)

                # Generate a trade signal
                if prediction == 1:  # Buy
                    action = "buy"
                elif prediction == 0:  # Sell
                    action = "sell"
                else:
                    action = None

            # Periodically optimize parameters using the genetic algorithm
            if time.time() - self.last_optimization_time >= 3600:  # Optimize every hour (adjust as needed)
                self.optimize_parameters_genetic()
                self.last_optimization_time = time.time()

            # Generate a trade signal
            if action is not None:
                current_time = time.time()
                entry_time = current_time  # Entry time at the beginning of the minute
                entry_price = self.candlestick_data[-1][4]  # Get the current close price

                # Calculate stop-loss and take-profit prices
                stop_loss_price = entry_price * (1 - self.stop_loss_percentage)
                take_profit_price = entry_price * (1 + self.take_profit_percentage)

                # Calculate position size
                position_size = self.balance * self.position_size_percentage

                signal = trade_signal.TradeSignal(entry_time, action, timeframe, "Pattern Recognition Strategy",
                                                stop_loss_price=stop_loss_price,
                                                take_profit_price=take_profit_price,
                                                position_size=position_size)
                self.pending_signals.append(signal)
                file_utils.save_signal_to_csv(signal.__dict__)
                self.data_manager.save_signals(self.pending_signals)
                print(f"Generated signal: {signal.__dict__}")

        self.save_strategies()
    def _calculate_sentiment_index(self, news_articles):
        """Calculates the overall sentiment index from the news articles."""
        return self.data_manager._calculate_sentiment_index(news_articles)

    def _analyze_sentiment_trends(self, articles):
        """Analyzes sentiment trends for each cluster."""
        cluster_sentiments = defaultdict(list)
        for article in articles:
            cluster_id = article['cluster_id']
            text = article['text']
            sentiment = self.analyze_sentiment(text)  # Use the new analyze_sentiment function
            cluster_sentiments[cluster_id].append(sentiment)

    def identify_candlestick_patterns(self, preprocessed_data):
        """Identifies candlestick patterns using k-means clustering."""
        features = np.array([data[0] for data in preprocessed_data])
        targets = np.array([data[1] for data in preprocessed_data])

        # Normalize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform k-means clustering
        clusters = self.kmeans.fit_predict(scaled_features)

        # Group candlestick data by cluster
        clustered_data = defaultdict(list)
        for i, cluster in enumerate(clusters):
            clustered_data[cluster].append((features[i], targets[i]))

        return clustered_data

    def train_prediction_model(self, clustered_data):
        """Trains a prediction model for each cluster of candlestick patterns."""
        self.prediction_models = {}
        for cluster_id, data in clustered_data.items():
            features, targets = zip(*data)
            model = LogisticRegression(C=self.logistic_regression_C)  # use the logistic_regression_C
            model.fit(features, targets)
            self.prediction_models[cluster_id] = model
        print("Prediction models trained!")

    def determine_timeframe(self, cluster_id, features):
        """Determines the optimal timeframe for a trade based on historical data."""
        # Analyze historical data for this cluster to find the average duration of price movements
        cluster_data = self.clustered_data[cluster_id]
        durations = []
        for i in range(len(cluster_data) - 1):
            current_features, target = cluster_data[i]
            next_features, next_target = cluster_data[i + 1]
            if target == 1:  # Price went up
                duration = next_features[0] - current_features[0]  # Calculate the duration of the price movement
                durations.append(duration)
        if durations:
            average_duration = np.mean(durations)
            timeframe = int(self.timeframe_multiplier * average_duration / 60)  # Convert to minutes (adjust as needed)
            return timeframe
        else:
            return 3   # Default timeframe (adjust as needed)

    def backtest_strategy(self, window_size=10, n_clusters=5, logistic_regression_C=1.0, timeframe_multiplier=1.0):
        """A backtesting strategy class for the beast."""
        self.dataclose = self.datas[0].close  # This line might be unused now
        self.preprocessed_data = []
        self.clustered_data = None
        self.prediction_models = {}
        self.kmeans = None
        self.params.window_size = window_size
        self.logistic_regression_C = logistic_regression_C
        self.timeframe_multiplier = timeframe_multiplier

        # Initialize KMeans model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # Adjust the number of clusters as needed

        # Use the price history from the data_manager
        data = pd.DataFrame(self.data_manager.price_history)
        data.columns = ['Close']  # Set the column name to 'Close'

        # Calculate technical indicators
        short_sma = data['Close'].rolling(window=self.short_sma_window).mean()
        long_sma = data['Close'].rolling(window=self.long_sma_window).mean()
        rsi = self.calculate_rsi(data['Close'], period=14) 
        macd, macdsignal, macdhist = self.calculate_macd(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Generate trading signals based on multiple indicators
        signals = pd.DataFrame(index=data.index)
        signals['sma_signal'] = np.where(short_sma > long_sma, 1, 0)
        signals['rsi_signal'] = np.where(rsi > 70, 0, np.where(rsi < 30, 1, 0))
        signals['macd_signal'] = np.where(macd > macdsignal, 1, 0)

        # Combine signals using a voting system (adjust weights as needed)
        signals['combined_signal'] = (signals['sma_signal'] + signals['rsi_signal'] + signals['macd_signal']) / 3

        # Generate trade signals based on the combined signal
        signals['buy_signal'] = np.where(signals['combined_signal'].shift(1) == 0 & signals['combined_signal'] == 1, 1, 0)
        signals['sell_signal'] = np.where(signals['combined_signal'].shift(1) == 1 & signals['combined_signal'] == 0, 1, 0)

        # Evaluate signals and calculate performance metrics
        total_signals = signals['buy_signal'].sum() + signals['sell_signal'].sum()
        wins = signals['buy_signal'][signals['Close'].shift(-1) > signals['Close']].sum() + signals['sell_signal'][signals['Close'].shift(-1) < signals['Close']].sum()
        losses = signals['buy_signal'][signals['Close'].shift(-1) < signals['Close']].sum() + signals['sell_signal'][signals['Close'].shift(-1) > signals['Close']].sum()
        win_rate = wins / total_signals if total_signals > 0 else 0

        # Calculate profit/loss
        profit_loss = (signals['buy_signal'] * (signals['Close'].shift(-1) - signals['Close'])).sum() + (signals['sell_signal'] * (signals['Close'] - signals['Close'].shift(-1))).sum()

        # Calculate drawdown
        drawdown = (data['Close'] / data['Close'].cummax() - 1).min()

        # Print performance metrics
        print(f"Total Signals: {total_signals}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Profit/Loss: {profit_loss:.2f}")
        print(f"Max Drawdown: {drawdown:.2f}")

        # Create a Backtrader cerebro instance
        cerebro = bt.Cerebro()

        # Add data to Backtrader
        data = bt.feeds.PandasData(dataname=pd.DataFrame(self.data_manager.price_history),
                                  fromdate=datetime.datetime(2023, 1, 1),  # Adjust date range as needed
                                  todate=datetime.datetime(2024, 1, 1))
        cerebro.adddata(data)

        # Add the trading strategy
        cerebro.addstrategy(self.BacktestStrategy, window_size=window_size, n_clusters=n_clusters,
                            logistic_regression_C=logistic_regression_C, timeframe_multiplier=timeframe_multiplier)

        # Set cash and commission
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)

        # Run the backtest
        results = cerebro.run()

        # Print backtest results
        print(f"Backtest Results: {results[0].analyzers.returns.get_analysis()}")

        return results

    def optimize_parameters(self):
        """Optimizes parameters based on backtest results."""
        # Define the search space for hyperparameters
        space = {
            'window_size': hp.choice('window_size', range(5, 21)),
            'n_clusters': hp.choice('n_clusters', range(3, 11)),
            'logistic_regression_C': hp.loguniform('logistic_regression_C', np.log(0.01), np.log(10)),
            'timeframe_multiplier': hp.uniform('timeframe_multiplier', 0.5, 2.0)
        }

        # Define the objective function
        def objective(params):
            # Backtest with the given parameters
            signals = self.backtest_strategy(params['window_size'], params['n_clusters'], params['logistic_regression_C'], params['timeframe_multiplier'])
            # Calculate a score based on the backtest results - you'll need to define a suitable score
            score = self.calculate_backtest_score(signals) 
            return {'loss': -score, 'status': STATUS_OK}

        # Perform hyperparameter optimization
        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

        # Update the beast's parameters
        self.params.window_size = best_params['window_size']
        self.kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=0)  # Reinitialize k-means with the new number of clusters
        self.logistic_regression_C = best_params['logistic_regression_C']
        self.timeframe_multiplier = best_params['timeframe_multiplier']

        print(f"Optimized parameters: {best_params}")
    def calculate_backtest_score(self, signals):
        """Calculates a score based on backtest signals. You'll need to define your own logic here."""
        # Example: Simple profit calculation
        profit = (signals['buy_signal'] * (signals['Close'].shift(-1) - signals['Close'])).sum() + (signals['sell_signal'] * (signals['Close'] - signals['Close'].shift(-1))).sum()
        return profit
    
    def calculate_rsi(self, prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        fast_ma = prices.rolling(window=fastperiod).mean()
        slow_ma = prices.rolling(window=slowperiod).mean()
        macd = fast_ma - slow_ma
        macdsignal = macd.rolling(window=signalperiod).mean()
        macdhist = macd - macdsignal
        return macd, macdsignal, macdhist

    def report_progress(self):
        """Reports the model's progress and statistics."""
        print("\n--- Model Progress Report ---")
        
        # Report model accuracy
        print(f"Model Accuracy: {self.accuracy:.2f}%")
        
        # Report learning progress (example assuming Keras model)
        if hasattr(self.model, 'history'):
            history = self.model.history.history
            if history:
                print(f"Last Epoch Loss: {history['loss'][-1]:.4f}")
                print(f"Last Epoch Accuracy: {history['accuracy'][-1]:.4f}")
        
        # Report trade outcomes
        self.report_trade_outcomes()
        
        # Report real-time predictions (example)
        if self.predictions:
            last_prediction = self.predictions[-1]
            print(f"Last Prediction: {last_prediction}")
            # Compare with actual outcome if available
        
        print("---------------------------------\n")

    def report_trade_outcomes(self):
        """Reports the outcomes of completed trades."""
        if self.completed_trades:
            total_profit = sum(trade.profit for trade in self.completed_trades)
            win_count = sum(trade.profit > 0 for trade in self.completed_trades)
            loss_count = len(self.completed_trades) - win_count
            win_ratio = win_count / len(self.completed_trades) if len(self.completed_trades) > 0 else 0.0
            print(f"Total Trades: {len(self.completed_trades)}")
            print(f"Wins: {win_count}, Losses: {loss_count}")
            print(f"Win Ratio: {win_ratio:.2f}")
            print(f"Total Profit: {total_profit:.2f}")
        else:
            print("No trades completed yet.")

    def build_cnn_lstm_attention_model(self, input_shape):
        """Builds the CNN-LSTM-Attention model."""
        # Define the input shape
        input_shape = (input_shape[1], input_shape[2])  # Adjust as needed

        # Define the input layer
        inputs = Input(shape=input_shape)

        # CNN layers
        x = Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)(inputs)
        x = MaxPooling1D(pool_size=2)(x)

        # LSTM layer
        x = LSTM(128, return_sequences=True)(x)

        # Split the LSTM output into query, value, and key tensors
        query = Lambda(lambda x: x[:, :, :64])(x)
        value = Lambda(lambda x: x[:, :, 64:128])(x)
        key = Lambda(lambda x: x[:, :, 128:])(x)

        # Apply the Attention layer
        attention_output = Attention()([query, value, key])

        # Flatten the output
        x = Flatten()(attention_output)

        # Dense layers
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification (buy/sell)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def train_cnn_lstm_attention_model(self):
            """Trains the CNN-LSTM-Attention model."""
            if len(self.data_manager.price_history) >= self.minimum_candlesticks:
                preprocessed_data = self.data_manager.preprocess_candlestick_data(
                    self.data_manager.price_history, window_size=self.params['window_size']
                )
                features = np.array([data[0] for data in preprocessed_data])
                targets = np.array([data[1] for data in preprocessed_data])

                # Reshape features for the CNN-LSTM model
                features = features.reshape(features.shape[0], features.shape[1], 1)

                self.cnn_lstm_attention_model.fit(features, targets, epochs=10, batch_size=32)  # Adjust epochs and batch size as needed
                print("CNN-LSTM-Attention model trained!")
    def evaluate_chromosome(self, chromosome):
        """Evaluates the fitness of a chromosome (set of parameters)."""
        window_size = int(chromosome[0])
        n_clusters = int(chromosome[1])
        logistic_regression_C = chromosome[2]
        timeframe_multiplier = chromosome[3]

        # Backtest with the given parameters
        signals = self.backtest_strategy(window_size, n_clusters, logistic_regression_C, timeframe_multiplier)

        # Calculate fitness score (e.g., Sharpe ratio)
        fitness = self.calculate_backtest_score(signals)  # Use your backtest score calculation

        return fitness,
    
    def optimize_parameters_genetic(self):
        """Optimizes parameters using a genetic algorithm."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_window_size", random.randint, 5, 20)
        toolbox.register("attr_n_clusters", random.randint, 3, 10)
        toolbox.register("attr_logistic_regression_C", random.uniform, 0.01, 10.0)
        toolbox.register("attr_timeframe_multiplier", random.uniform, 0.5, 2.0)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_window_size, toolbox.attr_n_clusters,
                          toolbox.attr_logistic_regression_C, toolbox.attr_timeframe_multiplier), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate_chromosome)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=50)  # Population size
        NGEN = 50  # Number of generations

        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))

        best_individual = tools.selBest(population, k=1)[0]

        # Update model parameters
        self.params['window_size'] = int(best_individual[0])
        self.params['n_clusters'] = int(best_individual[1])
        self.params['logistic_regression_C'] = best_individual[2]
        self.params['timeframe_multiplier'] = best_individual[3]

        print(f"Best parameters found: {self.params}")
    def analyze_sentiment(self, text):
        """Analyzes the sentiment of a given text using BERT/FinBERT."""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        # Map predicted class to sentiment (e.g., 0 - Negative, 1 - Positive)
        sentiment = {0: "negative", 1: "positive"}[predicted_class]
        return sentiment

    def tune_parameters(self):
        """Tunes the model's parameters using a grid search or other optimization techniques."""
        param_grid = {
            'window_size': [5, 10, 15, 20],
            'n_clusters': [3, 5, 7, 10],
            'logistic_regression_C': [0.1, 1.0, 10.0],
            'timeframe_multiplier': [0.5, 1.0, 1.5],
        }

        best_params = None
        best_score = -np.inf

        for params in ParameterGrid(param_grid):
            # Run backtest with the current parameters
            results = self.backtest_strategy(**params)

            # Evaluate the performance (e.g., Sharpe ratio)
            score = results[0].analyzers.returns.get_analysis()['sharpe']

            # Update best parameters if the score is better
            if score > best_score:
                best_score = score
                best_params = params

        # Update the model's parameters with the best values
        self.params.update(best_params)
        print(f"Best parameters found: {best_params}")

    def optimize_strategy(self):
        """Optimizes the trading strategy by experimenting with different combinations of indicators and rules."""
        # Define a set of indicators and rules to experiment with
        indicators = [
            ('SMA', self.calculate_sma),
            ('RSI', self.calculate_rsi),
            ('MACD', self.calculate_macd),
            # Add more indicators as needed
        ]

        rules = [
            ('Crossover', self.crossover_rule),
            ('RSI Divergence', self.rsi_divergence_rule),
            # Add more rules as needed
        ]

        best_strategy = None
        best_score = -np.inf

        for indicator_combination in itertools.combinations(indicators, 2):
            for rule_combination in itertools.combinations(rules, 1):
                # Create a new strategy using the current indicator and rule combinations
                strategy = self.create_strategy(indicator_combination, rule_combination)

                # Run backtest with the current strategy
                results = self.backtest_strategy(strategy=strategy)

                # Evaluate the performance (e.g., Sharpe ratio)
                score = results[0].analyzers.returns.get_analysis()['sharpe']

                # Update best strategy if the score is better
                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        # Update the model's strategy with the best values
        self.strategy = best_strategy
        print(f"Best strategy found: {best_strategy}")

    def evaluate_performance(self):
        """Evaluates the performance of the trading beast using backtesting and other metrics."""
        results = self.backtest_strategy()

        # Analyze backtesting results
        print(f"Backtest Results: {results[0].analyzers.returns.get_analysis()}")

        # Calculate performance metrics
        sharpe_ratio = results[0].analyzers.returns.get_analysis()['sharpe']
        drawdown = results[0].analyzers.drawdown.get_analysis()['max']
        win_rate = results[0].analyzers.trades.get_analysis()['won'] / results[0].analyzers.trades.get_analysis()['total']

        # Print performance metrics
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {drawdown:.2f}")
        print(f"Win Rate: {win_rate:.2f}")

        # Visualize the results (e.g., using matplotlib)
        plt.figure(figsize=(12, 6))
        plt.plot(results[0].datas[0].close, label='Close Price')
        plt.plot(results[0].datas[0].close[results[0].buy], '^', markersize=10, color='green', label='Buy')
        plt.plot(results[0].datas[0].close[results[0].sell], 'v', markersize=10, color='red', label='Sell')
        plt.title('Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def rsi_divergence_rule(self, rsi):
        """A rule that generates a buy signal when RSI shows bullish divergence."""
        # Check for bullish divergence
        if rsi[-1] > rsi[-2] and rsi[-2] < rsi[-3]:  # RSI is increasing
            if self.dataclose[-1] < self.dataclose[-2] and self.dataclose[-2] > self.dataclose[-3]:  # Price is decreasing
                return 1  # Buy signal
        return 0  # No signal