import websocket
import json
import collections
import time
import pickle
import csv
import os
import datetime

class CoinMarketCapWebsocketClient:
  """Connects to CoinMarketCap websocket and receives price updates."""

  def __init__(self, on_message_callback, on_error_callback=None, on_close_callback=None, 
               price_history_length=120, checkpoint_interval=60,  symbol='BTCUSDT'):
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
    self.price_data_file = "price_data.csv"  # File to store price data
    self.price_data_header = ['Date', 'Close']  
    self.message_buffer = ""
    self.symbol = symbol
    self.candlestick_data = collections.deque(maxlen=120)
    self.completed_candlesticks_file = "completed_candlesticks.csv"  # File to store completed candlesticks
    self._create_candlestick_csv()
    self.last_candle_start_time = None
    self.current_candle_high = None
    self.current_candle_low = None
    # Create the price data file and write the header if it doesn't exist
    if not os.path.exists(self.price_data_file):
        with open(self.price_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.price_data_header)

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
        """Handles incoming websocket messages, accumulating fragments until a complete JSON is received."""
        self.message_buffer += message
        # Attempt to parse the buffer as JSON
        try:
            data = json.loads(message)

            # Reset the buffer if parsing is successful
            self.message_buffer = ""
            timestamp = 0
            price = 0
            date_time = 0

            # Filter messages by symbol
            if data.get("d") and data["d"]["id"] == 1:
              price = data["d"]["p"]
              timestamp = int(data["t"]) / 1000  # Convert milliseconds to seconds
              date_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

              # Append price data to the CSV file
              with open(self.price_data_file, 'a', newline='') as f:
                  writer = csv.writer(f)
                  writer.writerow([date_time, price])
            self._update_candlestick_data(timestamp, price)
            print("candles", self.candlestick_data)
            # Call the user-defined on_message callback
            if self.on_message_callback:
                self.on_message_callback(message)  # Pass the parsed data

        except json.JSONDecodeError:
            print("json parsing error")
            pass
        except Exception as e:
           print("Error", e)

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

  def update_price_history(self, timestamp, price):
    """Updates the price history and handles checkpointing."""

    # Resumption Logic:
    last_processed_timestamp = self.load_checkpoint()
    if last_processed_timestamp and timestamp <= last_processed_timestamp:
      print(f"Skipping duplicate timestamp: {timestamp}")
      return  # Skip this message

    # Candlestick Logic (1-minute candles):
    if self.current_candle is None or timestamp >= self.current_candle[0] + 60:  # New candle every 60 seconds
        print(f"Creating new candle at timestamp: {timestamp}")
        # Create a new candle:
        if self.current_candle is not None:
          self.price_history.append(self.current_candle)  # Append the completed candle
          print(f"Appended completed candle: {self.current_candle}")
        self.current_candle = (timestamp, price, price, price, price)  # Open, High, Low, Close
    else:
        # Update the current candle:
        print(f"Updating current candle at timestamp: {timestamp}")
        self.current_candle = (
            self.current_candle[0],  # Open time
            max(self.current_candle[1], price),  # High
            min(self.current_candle[2], price),  # Low
            price  # Close
        )

    self.save_data(self.price_history)

    # Checkpointing
    if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
      self.save_checkpoint(timestamp)
      self.last_checkpoint_time = time.time()

  def _update_candlestick_data(self, timestamp, price):
        """Updates the candlestick data for the 1-minute timeframe."""
        # Align candle start times
        current_minute = datetime.datetime.fromtimestamp(timestamp).minute
        if self.last_candle_start_time is None or current_minute != self.last_candle_start_time:
            # New minute has begun, save the previous candle (if it exists)
            if self.candlestick_data:
                print("runned", self.candlestick_data[0])
                self._save_completed_candlestick(self.candlestick_data[0])
                self.current_candle_high = None
                self.current_candle_low = None

            self.last_candle_start_time = current_minute
            # No need to clear candlestick_data here, as it will automatically be overwritten when a new candle is created

        if not self.candlestick_data:
            # Create a new candle if the deque is empty
            new_candle = (timestamp, price, price, price, price)  # Open, High, Low, Close
            self.candlestick_data.append(new_candle)
        else:
            # Update the current candle
            last_candle = self.candlestick_data[-1]
            self.candlestick_data[-1] = (
                last_candle[0],  # Open time
                max(last_candle[1], price) if self.current_candle_high is None else max(self.current_candle_high, price),  # High
                min(last_candle[2], price) if self.current_candle_low is None else min(self.current_candle_low, price),  # Low
                price  # Close
            )

        # Update high and low prices for the current candle
        if self.current_candle_high is None or price > self.current_candle_high:
            self.current_candle_high = price
        if self.current_candle_low is None or price < self.current_candle_low:
            self.current_candle_low = price

  def _remove_incomplete_candles(self):
        """Removes incomplete candles from the CSV file that are older than 1 minute."""
        current_time = time.time()
        with open(self.completed_candlesticks_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            candlesticks = list(reader)
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
            for candlestick in candlesticks:
                entry_time = datetime.datetime.strptime(candlestick[0], '%Y-%m-%d %H:%M:%S').timestamp()
                if current_time - entry_time < 60:
                    writer.writerow(candlestick)

  def _save_completed_candlestick(self, candle):
        """Saves a completed candlestick to the CSV file."""
        entry_time, open_price, high_price, low_price, close_price = candle
        exit_time = entry_time + 60  # Exit time is 1 minute after entry
        with open(self.completed_candlesticks_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.datetime.fromtimestamp(entry_time).strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.datetime.fromtimestamp(exit_time).strftime('%Y-%m-%d %H:%M:%S'),
                    open_price,
                    close_price,
                    self.current_candle_high,  # Save the high price
                    self.current_candle_low,  # Save the low price
                ]
            )

        # Remove incomplete candles older than 1 minute
        self._remove_incomplete_candles()


  def _create_candlestick_csv(self):
        """Creates the CSV file for completed candlesticks with headers."""
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