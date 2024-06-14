import curses
import datetime
import pytz
from data_pipeline import data_manager
from model import trading_model
import time
import os
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
def main(stdscr):
    """The main function that orchestrates the trading AI."""
    # Initialize curses
    curses.noecho()  # Don't echo key presses
    curses.cbreak()  # React to key presses immediately
    stdscr.keypad(True)  # Enable arrow keys

    # Initialize Data Manager:
    data_mgr = data_manager.DataManager(**DATA_MANAGER_CONFIG)

    # Load or Create Trading Model:
    model_file = "trading_model.pkl"
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        try:
            model = trading_model.TradingModel(data_mgr)
            model.load_model_state()
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
            display_performance_report(stdscr, model)

            # Periodically save the model and trade history
            if time.time() % 600 < 1:  # Save every 10 minutes
                model.save_model_state()
                print("Model and trade history saved.")

            # Check for key presses
            key = stdscr.getch()
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping the trading AI...")

    finally:
        # Stop Websocket Feed:
        model.stop_websocket_feed()

        # Save the Model and Trade History:
        model.save_model_state()
        print("Trading model and trade history saved.")

    
def display_performance_report(stdscr, model, mse=None):
    """Displays a dynamic performance report of the trading model."""
    stdscr.clear()  # Clear the screen
    max_y, max_x = stdscr.getmaxyx() # Get terminal size

    # ---- Display Report in Fixed Rows ----
    row = 0
    stdscr.addstr(row, 0, "----- Performance Report -----")
    row += 1

    # Model Accuracy
    stdscr.addstr(row, 0, f"Model Accuracy: {model.accuracy:.2f}%")
    row += 1

    # Learning Progress (Example assuming Keras model)
    if hasattr(model.model, 'history'):
        history = model.model.history.history
        if history:
            stdscr.addstr(row, 0, f"Last Epoch Loss: {history['loss'][-1]:.4f}")
            row += 1
            stdscr.addstr(row, 0, f"Last Epoch Accuracy: {history['accuracy'][-1]:.4f}")
            row += 1

        # Trade Outcomes
    stdscr.addstr(row, 0, "----- Trade Outcomes -----")
    row += 1
    if model.completed_trades:
        total_profit = sum(trade.profit for trade in model.completed_trades)
        win_count = sum(trade.profit > 0 for trade in model.completed_trades)
        loss_count = len(model.completed_trades) - win_count
        win_ratio = win_count / len(model.completed_trades) if len(model.completed_trades) > 0 else 0.0
        stdscr.addstr(row, 0, f"Total Trades: {len(model.completed_trades)}")
        row += 1
        stdscr.addstr(row, 0, f"Wins: {win_count}, Losses: {loss_count}")
        row += 1
        stdscr.addstr(row, 0, f"Win Ratio: {win_ratio:.2f}")
        row += 1
        stdscr.addstr(row, 0, f"Total Profit: {total_profit:.2f}")
        row += 1
    else:
        stdscr.addstr(row, 0, "No trades completed yet.")
        row += 1

    # Real-time Predictions (Example)
    if model.predictions:
        last_prediction = model.predictions[-1]
        stdscr.addstr(row, 0, f"Last Prediction: {last_prediction}")
        row += 1
        # Compare with actual outcome if available

    # Other Existing Information
    executed_trades = len(model.trade_history)
    waiting_trades = len(model.pending_signals)
    completed_trades = len(
        [t for t in model.trade_history if t["status"] == "completed"]
    )
    winning_trades = len(
        [t for t in model.trade_history if t.get("result") == "win"]
    )
    losing_trades = len([t for t in model.trade_history if t.get("result") == "loss"])

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
        entry_time_tunisia = datetime.datetime.fromtimestamp(signal.entry_time, pytz.timezone('Africa/Tunis')).strftime('%Y-%m-%d %H:%M:%S')
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


if __name__ == "__main__":
    curses.wrapper(main)