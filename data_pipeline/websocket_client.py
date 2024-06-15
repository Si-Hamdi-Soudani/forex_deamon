import websocket
import json
import collections
import time
import pickle
import csv
import os
import datetime
import threading
from queue import Queue

class CoinMarketCapWebsocketClient:
    """Connects to CoinMarketCap websocket and receives price updates."""

    def __init__(self, on_message_callback, on_error_callback=None, on_close_callback=None, 
                 price_history_length=120, checkpoint_interval=60, symbol='BTCUSDT'):
        """Initializes the websocket client."""
        self.on_message_callback = on_message_callback
        self.on_error_callback = on_error_callback
        self.on_close_callback = on_close_callback
        self.price_history_length = price_history_length
        self.checkpoint_interval = checkpoint_interval
        self.price_history = collections.deque(maxlen=self.price_history_length)
        self.last_checkpoint_time = time.time()
        self.current_candle = None
        self.ws = None
        self.connected = False
        self.symbol = symbol
        self.candlestick_data = collections.deque(maxlen=120)
        self.completed_candlesticks_file = "completed_candlesticks.csv"  # File to store completed candlesticks
        self._create_candlestick_csv()
        self.last_candle_start_time = None
        self.current_candle_high = None
        self.current_candle_low = None 
        self.validate_latest_candle()


    def on_open(self, ws):
        """Sends the subscription message when the connection is opened."""
        ws.send(json.dumps({
            "method": "RSUBSCRIPTION",
            "params": [
                "main-site@crypto_price_5s@{}@normal",
                "1"
            ]
        }))

    def on_message(self, ws, message):
        """Handles incoming websocket messages."""
        try:
            data = json.loads(message)
            # Filter messages by symbol
            if data.get("d") and data["d"]["id"] == 1:
                price = data["d"]["p"]
                timestamp = int(data["t"]) / 1000  # Convert milliseconds to seconds

                # Print received data for debugging
                #print(f"Received data - timestamp: {timestamp}, price: {price}")

                # Update candlestick data and get the completed candle
                completed_candle = self._update_candlestick_data(timestamp, price) 

                # Pass the completed candle to the DataManager
                if completed_candle:
                    #print(f"Completed candle: {completed_candle}")
                    self.on_message_callback(completed_candle) 

        except json.JSONDecodeError as e:
            print("JSON parsing error", str(e))
        except Exception as e:
            print("Error:", str(e))

    def on_error(self, ws, error):
        """Handles websocket errors."""
        if self.on_error_callback:
            self.on_error_callback(error)

    def on_close(self, ws, close_status_code, close_msg):
        """Handles websocket closure."""
        if self.on_close_callback:
            self.on_close_callback(close_status_code, close_msg)


    def connect(self):
        """Connects to the CoinMarketCap websocket."""
        self.ws = websocket.WebSocketApp(
            "wss://push.coinmarketcap.com/ws?device=web&client_source=coin_detail_page",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            header=[
                "Accept-Encoding: gzip, deflate, br, zstd",
                "Accept-Language: fr-FR,fr;q=0.9",
                "Cache-Control: no-cache",
                "Connection: Upgrade",
                "Host: push.coinmarketcap.com",
                "Origin: https://websocketking.com", 
                "Pragma: no-cache",
                "Sec-Websocket-Extensions: permessage-deflate; client_max_window_bits",
                "Sec-Websocket-Key: zAN8N9C+YFPmDvBh0iVvhA==", 
                "Sec-Websocket-Version: 13",
                "Upgrade: websocket",
                "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0"
            ]
        )
        self.connected = True
        self.ws.run_forever(ping_interval=10, ping_timeout=5)

    def disconnect(self):
        """Disconnects from the websocket."""
        if self.ws:
            self.ws.close()
            self.connected = False

    def save_data(self, data, filename='data.pkl'):
        """Saves data to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, filename='data.pkl'):
        """Loads data from a pickle file."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        

    def save_checkpoint(self, timestamp, filename='checkpoint.pkl'):
        """Saves a checkpoint with the last processed timestamp."""
        with open(filename, 'wb') as f:
            pickle.dump(timestamp, f)

    def load_checkpoint(self, filename='checkpoint.pkl'):
        """Loads the last processed timestamp from a checkpoint."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def _update_candlestick_data(self, timestamp, price):
        """Updates the candlestick data and returns the completed candle (if any)."""
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        current_minute = dt_object.minute
        current_second = dt_object.second

        completed_candle = None  # Initialize completed_candle

        # Check if we need to start a new candle
        if self.last_candle_start_time is None or (current_minute != self.last_candle_start_time and current_second == 2):
            # Save the previous candle if it exists
            if self.candlestick_data:
                completed_candle = self.candlestick_data[-1]  # Assign the completed candle
                self._save_completed_candlestick(completed_candle)
                self.candlestick_data.popleft()

            # Start a new candle (as a dictionary)
            self.candlestick_data.append({
                'entry_time': timestamp,
                'exit_time': timestamp,
                'open_price': price,
                'close_price': price,
                'high_price': price,
                'low_price': price,
            })
            self.current_candle_high = price
            self.current_candle_low = price
            self.last_candle_start_time = current_minute
        else:
            # Update the current candle
            if self.candlestick_data:
                last_candle = self.candlestick_data[-1]
                self.current_candle_high = max(self.current_candle_high, price)
                self.current_candle_low = min(self.current_candle_low, price)
                self.candlestick_data[-1] = {
                    'entry_time': last_candle['entry_time'],
                    'exit_time': timestamp,
                    'open_price': last_candle['open_price'],
                    'close_price': price,
                    'high_price': self.current_candle_high,
                    'low_price': self.current_candle_low,
                }
        return completed_candle  # Return the completed candle (if any)

    def _save_completed_candlestick(self, candle):
        """Saves a completed candlestick to the CSV file."""
        #print("Saving candle:", candle)
        with open(self.completed_candlesticks_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.datetime.fromtimestamp(candle['entry_time']).strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.datetime.fromtimestamp(candle['exit_time']).strftime('%Y-%m-%d %H:%M:%S'),
                    candle['open_price'], 
                    candle['close_price'],
                    candle['high_price'],
                    candle['low_price'],
                ]
            )

    def _create_candlestick_csv(self):
        """Creates the CSV file for completed candlesticks with headers."""
        if not os.path.exists(self.completed_candlesticks_file):
            with open(self.completed_candlesticks_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Entry Time",
                        "Exit Time",
                        "Open Price",
                        "Close Price",
                        "High Price",
                        "Low Price",
                    ]
                )


    def validate_latest_candle(self):
        """Validates the latest candle in the CSV file and deletes it if it doesn't meet the criteria."""
        if not os.path.exists(self.completed_candlesticks_file):
            return

        with open(self.completed_candlesticks_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return  # No candles to validate

        latest_candle = lines[-1].strip().split(',')
        entry_time = datetime.datetime.strptime(latest_candle[0], '%Y-%m-%d %H:%M:%S')
        exit_time = datetime.datetime.strptime(latest_candle[1], '%Y-%m-%d %H:%M:%S')

        # Check if the candle starts at the second 2 and ends at the second 2 of the next minute
        if entry_time.second != 2 or (exit_time - entry_time).seconds != 60:
            print(f"Invalid latest candle: {latest_candle}. Deleting it.")
            with open(self.completed_candlesticks_file, 'w') as f:
                f.writelines(lines[:-1])  # Delete the last line