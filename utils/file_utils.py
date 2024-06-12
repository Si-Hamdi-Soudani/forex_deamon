import csv
import os

def save_signal_to_csv(signal, filename="signals.csv"):
    """Saves a trade signal to a CSV file.

    Args:
        signal: A dictionary containing the trade signal details.
        filename: The name of the CSV file.
    """
    with open(filename, "a", newline="") as csvfile:
        fieldnames = ['exit_time', 'exit_price', 'candles', 'entry_price', 'result', 'signal_id', 'strategy_name', 'timeframe', 'entry_time', 'action', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(signal)


def save_trade_to_csv(trade_data, filename="trade_history.csv"):
    """Saves trade data to a CSV file."""
    fieldnames = ['timestamp', 'price', 'action', 'timeframe', 
                  'status', 'entry_price', 'exit_time', 'exit_price', 
                  'result', 'signal_id', 'strategy_name', 'candles', 'trade_id', 'entry_time']  # Make sure all fields are here

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader() 

        writer.writerow(trade_data)