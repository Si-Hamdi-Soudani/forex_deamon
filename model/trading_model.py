import pickle
import threading
import time
import json
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Attention
from data_pipeline import websocket_client, data_manager
from trading_logic import trade_signal, trading_strategy
from utils import file_utils
import os
import logging
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import pandas as pd 
import tensorflow as tf

def ml_strategy_condition(price_data):
    """Condition for ML-based strategy."""
    # Placeholder condition: always return True
    # You can implement more complex conditions based on your requirements
    return True

class TradingModel:
    """The core trading model (the beast's brain)."""

    def __init__(self, data_manager, short_sma_window=10, long_sma_window=20, cooldown_period=60):
        """Initializes the trading model."""
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
        self.price_data = self._load_price_data()
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

        # Load model state on initialization
        self.load_model_state()

        # Initialize ensemble models
        self.ensemble_models = []

        logistic_model = LogisticRegression()
        random_forest_model = RandomForestClassifier()

        self.add_ensemble_model(logistic_model, "Logistic Regression")
        self.add_ensemble_model(random_forest_model, "Random Forest")

        self.sentiment_index = None

    @property
    def websocket_connected(self):
        """Returns True if the websocket is connected, False otherwise."""
        return self.websocket_client.connected if self.websocket_client else False

    def on_price_update(self, message):
        """Handles price updates from the websocket."""
        try:
            data = json.loads(message)
            if data.get("d") and data["d"]["id"] == 1:
                price = data["d"]["p"]
                timestamp = int(data["t"]) / 1000
                self.data_manager.update_price_history(timestamp, price)
                self.check_for_trade_entry(timestamp)
                self.track_active_trades(timestamp, price)
                self.evaluate_and_execute_trade()
        except Exception as e:
            print(f"Error processing price update: {e}")

    def on_error(self, error):
        """Handles websocket errors."""
        print(f"Websocket Error: {error} t")

    def on_close(self, code, msg):
        """Handles websocket closure."""
        print(f"Websocket Closed: {code} - {msg}")
    
    def check_for_trade_entry(self, timestamp):
        """Checks if it's time to enter new trades based on generated signals."""
        for signal in self.pending_signals:
            if timestamp >= signal.entry_time:
                signal.activate(self.data_manager.price_history[-1][1])  # Get the latest price
                self.active_trades.append(signal)
                logging.info(f"Entered trade: {signal.__dict__}")  # Use logging.info()
                self.pending_signals.remove(signal)

                # Update last trade time
                self.last_trade_time = timestamp

    def track_active_trades(self, timestamp, price):
        """Tracks active trades, records candle data, and checks for exit conditions."""
        for trade in self.active_trades:
            trade.add_candle(timestamp, price)
            time_elapsed = timestamp - trade.entry_time
            if time_elapsed >= trade.timeframe:
                self.exit_trade(trade, timestamp, price)

    def exit_trade(self, trade, timestamp, price):
        """Exits the active trade and logs the results."""
        trade.complete(timestamp, price)
        self.log_trade(trade)
        self.active_trades.remove(trade)

        # Update strategy performance
        for strategy in self.strategies:
            if strategy.name == trade.strategy_name:
                strategy.update_performance(trade.result)

        # **FIX: Update trade history**
        self.trade_history.append(trade.__dict__)

    def log_trade(self, trade):
        """Logs the trade details to a CSV file."""
        file_utils.save_trade_to_csv(trade.__dict__)
    
    def evaluate_and_execute_trade(self):
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

        self.data_manager.fetch_coindesk_news()
        self.data_manager.fetch_binance_news()
        self.data_manager.cluster_news_articles()  # Cluster articles
        news_articles = self.data_manager.get_news_articles()

        # Calculate sentiment index (ensure it's accessible to predict_ensemble)
        self.sentiment_index = self._calculate_sentiment_index(news_articles)
        print(f"Overall Sentiment Index: {self.sentiment_index}")

        # Analyze sentiment trends
        self._analyze_sentiment_trends(news_articles)

        # Evaluate ALL strategies (loaded and ML-based)
        for strategy in self.strategies:
            signal_data = strategy.evaluate(self.data_manager.price_history, self.sentiment_index)  # Pass sentiment index to strategies

            if signal_data is not None:
                current_minute = int(current_time // 60)

                if current_minute == last_minute and trades_this_minute >= max_trades_per_minute:
                    print(f"Reached maximum trades ({max_trades_per_minute}) for this minute. Skipping signal.")
                    continue

                # Check signal generation limit
                if self.signals_generated >= self.signal_generation_limit:
                    print(f"Signal generation limit ({self.signal_generation_limit}) reached. Skipping signal.")
                    continue

                current_time = time.time()
                entry_time = current_time + 180

                if "RSI" in strategy.name:
                    timeframe = 60
                else:
                    timeframe = 120

                signal = trade_signal.TradeSignal(entry_time, signal_data["action"], timeframe, strategy.name)
                self.pending_signals.append(signal)
                file_utils.save_signal_to_csv(signal.__dict__)
                self.data_manager.save_signals(self.pending_signals)

                trades_this_minute += 1
                self.last_trade_time = current_time
                self.signals_generated += 1  # Increment the signal generation count

        # Generate a new ML-based strategy (with conditions and actions) once every hour
        if current_time - self.last_strategy_generation_time >= 3600:
            if len(self.data_manager.price_history) >= self.short_sma_window:
                prices = [candle[1] for candle in self.data_manager.price_history]
                prediction = self.predict_ensemble(prices[-self.short_sma_window:])  # Use sentiment-weighted ensemble
                action = "buy" if prediction == 0 else "sell"

                new_strategy = trading_strategy.TradingStrategy(
                    f"ML Strategy {int(time.time())}",
                    conditions=[ml_strategy_condition],
                    actions=[action],
                    parameters={}
                )
                self.strategies.append(new_strategy)
                print(f"Generated new strategy: {new_strategy.name}, Action: {action}")

                self.last_strategy_generation_time = current_time

        self.save_strategies()

    def start_websocket_feed(self):
        """Starts the websocket client in a separate thread."""
        self.websocket_client = websocket_client.CoinMarketCapWebsocketClient(
            on_message_callback=self.on_price_update,
            on_error_callback=self.on_error,
            on_close_callback=self.on_close,
        )
        websocket_thread = threading.Thread(target=self.websocket_client.connect)
        websocket_thread.start()

    def stop_websocket_feed(self):
        """Stops the websocket client."""
        if self.websocket_client:
            self.websocket_client.disconnect()
    
    def save_model(self):
        """Saves the model to a file."""
        # Temporarily remove the websocket client for pickling
        websocket_client_backup = self.websocket_client
        self.websocket_client = None
        self.save_data(self, self.model_file)
        # Restore the websocket client
        self.websocket_client = websocket_client_backup

    def save_trade_history(self, filename="trade_history.pkl"):
        """Saves the trade history to a file."""
        self.save_data(self.trade_history, filename)

    def save_data(self, data, filename):
        """Saves data to a pickle file."""
        os.makedirs(self.data_manager.data_dir, exist_ok=True)  # Create data directory if it doesn't exist
        with open(os.path.join(self.data_manager.data_dir, filename), "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        """Loads data from a pickle file."""
        try:
            with open(os.path.join(self.data_manager.data_dir, filename), "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def load_model_state(self):
        """Loads the model state from a file."""
        model = self.load_data(self.model_file)
        if model:
            self.current_position = model.current_position
            self.trade_history = model.trade_history
            self.active_trades = model.active_trades
            self.pending_signals = model.pending_signals

            # FIX: Handle strategy compatibility during loading
            for i, strategy in enumerate(model.strategies):
                if not hasattr(strategy, "conditions") or not hasattr(strategy, "actions"):
                    # If strategy is missing conditions/actions, re-create it
                    if strategy.name == "SMA Crossover":
                        model.strategies[i] = trading_strategy.create_sma_crossover_strategy()  # Update the strategy in the list
                    elif strategy.name == "RSI Overbought":
                        model.strategies[i] = trading_strategy.create_rsi_overbought_strategy()  # Update the strategy in the list
                    else:
                        # Handle other strategy types or create a default
                        model.strategies[i] = trading_strategy.TradingStrategy(
                            strategy.name,
                            conditions=[lambda price_data: True],
                            actions=["buy"],
                            parameters=strategy.parameters
                        )  # Update the strategy in the list

            self.strategies = model.strategies

    @staticmethod
    def load_model(filename="trading_model.pkl"):
        """Loads the model from a file."""
        model = TradingModel(data_manager.DataManager())
        model.model_file = filename
        model.load_model_state()  # Load the model state
        model.start_websocket_feed()  # Reinitialize the websocket client
        return model

    def train(self, X, y):
        """Trains the model using historical data."""
        X = np.array(X).reshape((len(X), -1, 1))
        y = np.array(y)
        self.model.fit(X, y, epochs=10, batch_size=32)

    def predict(self, X):
        """Predicts the next action (buy/sell) based on price history."""
        X = np.array(X).reshape((1, -1, 1))
        prediction = self.model.predict(X)
        return np.argmax(prediction)

    def select_strategy(self):
        """Selects the best performing strategy based on win rate."""
        best_strategy = None
        best_win_rate = 0

        for strategy in self.strategies:
            win_rate = strategy.get_win_rate()
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_strategy = strategy

        return best_strategy
    
    def load_strategies(self, data_dir="data"):
        """Loads trading strategies from files, appending to existing."""
        for filename in os.listdir(data_dir):
            if filename.startswith("strategy_") and filename.endswith(".pkl"):
                strategy_id = filename[9:-4]
                strategy = trading_strategy.TradingStrategy.load(strategy_id, data_dir)
                if strategy:
                    self.strategies.append(strategy)
    
    def save_strategies(self):
        """Saves all trading strategies to files."""
        for strategy in self.strategies:
            strategy.save(self.data_manager.data_dir)
    
    def optimize_hyperparameters(self):
        """Optimizes hyperparameters using Bayesian optimization and genetic algorithms."""

        def objective(params):
            """Objective function for hyperparameter optimization."""
            model = Sequential([
                Conv1D(filters=int(params['filters']), kernel_size=int(params['kernel_size']), activation='relu', input_shape=(None, 1)),
                MaxPooling1D(pool_size=int(params['pool_size'])),
                LSTM(int(params['lstm_units']), return_sequences=True),
                Attention(),
                LSTM(int(params['lstm_units'])),
                Dense(1),
            ])
            model.compile(optimizer=params['optimizer'], loss='mse')
            model.fit(self.X_train, self.y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0)
            loss = model.evaluate(self.X_val, self.y_val, verbose=0)
            return {'loss': loss, 'status': STATUS_OK}

        # Define the search space
        space = {
            'filters': hp.quniform('filters', 32, 128, 1),
            'kernel_size': hp.quniform('kernel_size', 2, 5, 1),
            'pool_size': hp.quniform('pool_size', 2, 5, 1),
            'lstm_units': hp.quniform('lstm_units', 32, 128, 1),
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'epochs': hp.quniform('epochs', 10, 50, 1),
            'batch_size': hp.quniform('batch_size', 16, 64, 1),
        }

        # Run Bayesian optimization
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)
        best['filters'] = int(best['filters'])
        best['kernel_size'] = int(best['kernel_size'])
        best['pool_size'] = int(best['pool_size'])
        best['lstm_units'] = int(best['lstm_units'])
        best['epochs'] = int(best['epochs'])
        best['batch_size'] = int(best['batch_size'])
        best['optimizer'] = ['adam', 'rmsprop'][best['optimizer']]

        # Update model with the best hyperparameters
        self.model = Sequential([
            Conv1D(filters=best['filters'], kernel_size=best['kernel_size'], activation='relu', input_shape=(None, 1)),
            MaxPooling1D(pool_size=best['pool_size']),
            LSTM(best['lstm_units'], return_sequences=True),
            Attention(),
            LSTM(best['lstm_units']),
            Dense(1),
        ])
        self.model.compile(optimizer=best['optimizer'], loss='mse')
        print(f"Optimized hyperparameters: {best}")

    def genetic_algorithm_optimization(self):
        """Optimizes hyperparameters using genetic algorithms."""

        def evaluate_individual(individual):
            """Evaluates an individual's fitness."""
            filters, kernel_size, pool_size, lstm_units, optimizer, epochs, batch_size = individual
            model = Sequential([
                Conv1D(filters=int(filters), kernel_size=int(kernel_size), activation='relu', input_shape=(None, 1)),
                MaxPooling1D(pool_size=int(pool_size)),
                LSTM(int(lstm_units), return_sequences=True),
                Attention(),
                LSTM(int(lstm_units)),
                Dense(1),
            ])
            model.compile(optimizer=optimizer, loss='mse')
            model.fit(self.X_train, self.y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
            loss = model.evaluate(self.X_val, self.y_val, verbose=0)
            return loss,

        # Define the genetic algorithm parameters
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_int", np.random.randint, 32, 128)
        toolbox.register("attr_float", np.random.uniform, 2, 5)
        toolbox.register("attr_choice", np.random.choice, ['adam', 'rmsprop'])
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_int, toolbox.attr_float, toolbox.attr_float, toolbox.attr_int,
                          toolbox.attr_choice, toolbox.attr_int, toolbox.attr_int), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=32, up=128, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate_individual)

        # Run the genetic algorithm
        population = toolbox.population(n=50)
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
        best_individual = tools.selBest(population, k=1)[0]
        best_params = {
            'filters': int(best_individual[0]),
            'kernel_size': int(best_individual[1]),
            'pool_size': int(best_individual[2]),
            'lstm_units': int(best_individual[3]),
            'optimizer': best_individual[4],
            'epochs': int(best_individual[5]),
            'batch_size': int(best_individual[6]),
        }

        # Update model with the best hyperparameters
        self.model = Sequential([
            Conv1D(filters=best_params['filters'], kernel_size=best_params['kernel_size'], activation='relu', input_shape=(None, 1)),
            MaxPooling1D(pool_size=best_params['pool_size']),
            LSTM(best_params['lstm_units'], return_sequences=True),
            Attention(),
            LSTM(best_params['lstm_units']),
            Dense(1),
        ])
        self.model.compile(optimizer=best_params['optimizer'], loss='mse')
        print(f"Optimized hyperparameters using Genetic Algorithm: {best_params}")

    def add_ensemble_model(self, model, name):
        """Adds a model to the ensemble."""
        self.ensemble_models.append((name, model))

    def train_ensemble(self, X, y):
        """Trains the ensemble of models."""
        for name, model in self.ensemble_models:
            print(f"Training model: {name}")
            model.fit(X, y)

    def predict_ensemble(self, X):
        """Predicts the next action (buy/sell) using a sentiment-weighted ensemble of models."""
        predictions = []
        model_weights = []
        for name, model in self.ensemble_models:
            prediction = model.predict(X)
            predictions.append(prediction)

            # Calculate sentiment weight for the model (example: higher weight if prediction aligns with sentiment)
            if (prediction == 0 and self.sentiment_index > 0) or (prediction == 1 and self.sentiment_index < 0):
                sentiment_weight = 1.2  # Adjust weight as needed
            else:
                sentiment_weight = 0.8  # Adjust weight as needed
            model_weights.append(sentiment_weight)

        # Weighted average of predictions
        final_prediction = np.round(np.average(predictions, weights=model_weights))
        return final_prediction
    
    def _analyze_sentiment_trends(self, articles):
        """Analyzes sentiment trends for each cluster."""
        cluster_sentiments = defaultdict(list)
        for article in articles:
            cluster_id = article['cluster_id']
            sentiment = article['sentiment']
            cluster_sentiments[cluster_id].append(sentiment)

        for cluster_id, sentiments in cluster_sentiments.items():
            self.sentiment_history[cluster_id].append(sum(sentiments) / len(sentiments))  # Store average sentiment

            # Analyze trends (example: check for significant changes in recent sentiment)
            if len(self.sentiment_history[cluster_id]) > 5:  # Analyze trends over the last 5 time periods
                recent_sentiment = self.sentiment_history[cluster_id][-5:]
                sentiment_change = recent_sentiment[-1] - recent_sentiment[0]
                if sentiment_change > 0.2:
                    print(f"Significant positive sentiment shift detected in cluster {cluster_id}")
                elif sentiment_change < -0.2:
                    print(f"Significant negative sentiment shift detected in cluster {cluster_id}")
    
    def _calculate_cluster_score(self, cluster_id):
        """Calculates a score for a cluster based on sentiment, trends, and correlations."""
        sentiment_score = self.sentiment_history[cluster_id][-1]  # Latest sentiment for the cluster
        correlation_score = self._correlate_sentiment_with_price(cluster_id)
        # Add more scoring components as needed (e.g., trend score, event score)

        # Combine scores (example: weighted average)
        cluster_score = 0.5 * sentiment_score + 0.5 * correlation_score 
        return cluster_score
    
    def _load_price_data(self):
        """Loads historical price data from the CSV file."""
        try:
            price_data = pd.read_csv('price_data.csv', index_col='Date', parse_dates=True)
            return price_data
        except FileNotFoundError:
            print("Price data file not found. Ensure 'price_data.csv' exists.")
            return None