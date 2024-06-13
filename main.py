import curses
from data_pipeline import data_manager
from model import trading_model
import time
import os
import numpy as np
from datetime import datetime, timedelta
import pytz
import pickle

# --- Configuration ---
DATA_MANAGER_CONFIG = {
    "price_history_length": 120,
    "checkpoint_interval": 60,
}
TRADING_MODEL_CONFIG = {
    "short_sma_window": 10,
    "long_sma_window": 20,
}

# --- Main Function ---
def main():
    """The main function that orchestrates the trading AI."""
    # Initialize curses
    # curses.noecho()  # Don't echo key presses
    # curses.cbreak()  # React to key presses immediately
    # stdscr.keypad(True)  # Enable arrow keys

    # Initialize Data Manager:
    data_mgr = data_manager.DataManager(**DATA_MANAGER_CONFIG)

    # Load or Create Trading Model:
    model_file = "trading_model.pkl"
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        try:
            model = trading_model.TradingModel.load_model(model_file)
            print("Loaded existing trading model.")
        except (EOFError, pickle.UnpicklingError):
            print("Failed to load existing model. Creating a new one.")
            model = trading_model.TradingModel(data_mgr, **TRADING_MODEL_CONFIG)
    else:
        model = trading_model.TradingModel(data_mgr, **TRADING_MODEL_CONFIG)
        print("Created a new trading model.")

    # Start Websocket Feed (Connect the client to the model):
    model.start_websocket_feed()

    # Main Loop:
    try:
        while True:
            time.sleep(1)  # Check for new data every second
            # Display performance report
            #display_performance_report(stdscr, model)

            # Periodically train the model
            if time.time() % 600 < 1:  # Train every hour
                print("Training the model...")
                mse = train_model(model)  # Get MSE value from training

            # Save the model and trade history periodically
            if time.time() % 600 < 1:  # Save every 10 minutes
                model.save_model()
                print("Model and trade history saved.")
                            # Check for key presses
            # key = stdscr.getch()
            # if key == ord("q"):
            #     break

    except KeyboardInterrupt:
        print("Stopping the trading AI...")

    finally:
        # Stop Websocket Feed:
        model.stop_websocket_feed()

        # Save the Model and Trade History:
        model.save_model()
        print("Trading model and trade history saved.")

# --- Training Function ---
def train_model(model):
    """Trains the trading model using historical data."""
    # Prepare training data (X: input sequences, y: target prices)
    X = []
    y = []
    price_history = list(model.data_manager.price_history)
    for i in range(len(price_history) - model.short_sma_window):
        X.append(price_history[i : i + model.short_sma_window])
        y.append(
            price_history[i + model.short_sma_window][1]
        )  # Target is the next price

    if X and y:
        # Split data into training and validation sets
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        model.train(X_train, y_train)

        # Calculate MSE on validation set
        mse = model.model.evaluate(np.array(X_val).reshape((len(X_val), -1, 1)), np.array(y_val), verbose=0)
        print(f"Model trained. Validation MSE: {mse}")
        return mse  # Return the MSE value
    else:
        print("Not enough data to train the model.")
        return None

def display_performance_report(stdscr, model, mse=None):
    """Displays a dynamic performance report of the trading model."""
    executed_trades = len(model.trade_history)
    waiting_trades = len(model.pending_signals)
    completed_trades = len(
        [t for t in model.trade_history if t["status"] == "completed"]
    )
    winning_trades = len(
        [t for t in model.trade_history if t.get("result") == "win"]
    )
    losing_trades = len([t for t in model.trade_history if t.get("result") == "loss"])

    stdscr.clear()  # Clear the screen
    max_y, max_x = stdscr.getmaxyx() # Get terminal size

    # ---- Display Report in Fixed Rows ----
    row = 0
    stdscr.addstr(row, 0, "----- Performance Report -----")
    row += 1
    stdscr.addstr(row, 0, f"Executed Trades: {executed_trades}")
    row += 1
    stdscr.addstr(row, 0, f"Waiting Trades: {waiting_trades}")
    row += 1
    stdscr.addstr(row, 0, f"Completed Trades: {completed_trades}")
    row += 1
    stdscr.addstr(row, 0, f"Winning Trades: {winning_trades}")
    row += 1
    stdscr.addstr(row, 0, f"Losing Trades: {losing_trades}")
    row += 1
    stdscr.addstr(row, 0, f"Total Strategies: {len(model.strategies)}")
    row += 1
    if mse is not None:
        stdscr.addstr(row, 0, f"Model Validation MSE: {mse:.4f}")
    row += 1

    # Display Websocket Status
    websocket_status = "Connected" if model.websocket_connected else "Disconnected"
    stdscr.addstr(row, 0, f"Websocket: {websocket_status}")
    row += 1 

    stdscr.addstr(row, 0, "-----------------------------")
    row += 2
        # ---- Display Waiting Trades ----
    stdscr.addstr(row, 0, "----- Waiting Trades -----")
    row += 1
    for i, signal in enumerate(model.pending_signals):
        if row + i >= max_y: # Stop if we reach the bottom of the screen
            break
        entry_time_tunisia = datetime.fromtimestamp(signal.entry_time, pytz.timezone('Africa/Tunis')).strftime('%Y-%m-%d %H:%M:%S')
        stdscr.addstr(row + i, 0, f"Signal ID: {signal.signal_id}, Entry Time (Tunisia): {entry_time_tunisia}, Action: {signal.action}, Timeframe: {signal.timeframe}s, Strategy: {signal.strategy_name}")
    row = min(row + len(model.pending_signals) + 1, max_y - 1)
    stdscr.addstr(row, 0, "-----------------------------")

    # ---- Display Strategy Statistics ----
    row += 2  # Add some spacing
    stdscr.addstr(row, 0, "----- Strategy Statistics -----")
    row += 1
    for i, strategy in enumerate(model.strategies):
        if row + i >= max_y:
            break
        win_rate = strategy.get_win_rate() * 100
        stdscr.addstr(row + i, 0, f"Strategy: {strategy.name}, Wins: {strategy.performance['wins']}, Losses: {strategy.performance['losses']}, Win Rate: {win_rate:.2f}%")

    stdscr.refresh()  # Update the display

# --- Run the AI ---
if __name__ == "__main__":
    #curses.wrapper(main)  # Run the main function with curses
    main()