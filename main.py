# main.py

import threading
import time
from news_devourer import NewsDevourer
from chart_watcher import ChartWatcher
from trade_manager import TradeManager
from learning_daemon import LearningDaemon
from utils.helpers import load_config

# Load configuration settings
config = load_config("config.py")

# Initialize the demonic entities
news_devourer = NewsDevourer(config)
chart_watcher = ChartWatcher(config)
trade_manager = TradeManager(config)
learning_daemon = LearningDaemon(config)

# Create and start the threads
threads = [
    threading.Thread(target=news_devourer.start_consuming),
    threading.Thread(target=chart_watcher.start_watching),
    threading.Thread(target=trade_manager.start_trading),
    threading.Thread(target=learning_daemon.start_learning),
]

for thread in threads:
    thread.start()

# Keep the main thread alive (optional)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Handle graceful shutdown if needed
    print("Shutting down the demonic trading engine...")
    for thread in threads:
        thread.join()
    print("The beast slumbers... for now.")
