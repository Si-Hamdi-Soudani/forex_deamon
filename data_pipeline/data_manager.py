import collections
import csv
import time
import pickle
import os
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from trading_logic import trade_signal, trading_strategy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import datetime
import requests

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
        self.news_articles = []
        
        # Load data on initialization
        self.price_history = self.load_data(self.price_history_file) or collections.deque(maxlen=self.price_history_length)
        self.last_checkpoint_time = self.load_checkpoint(self.checkpoint_file) or time.time()
        self.pending_signals = self.load_signals(self.data_dir) or []

        self.word2vec_model = None  # Initialize model as None
        self.num_clusters = 5
        self.word2vec_model_file = os.path.join(self.data_dir, "word2vec_model.model")

        # Fetch news data first
        self.fetch_binance_news()
        self.fetch_coindesk_news()

        # Try to load the model, train if not found
        if os.path.exists(self.word2vec_model_file):
            print("Loading existing Word2Vec model...")
            self.word2vec_model = Word2Vec.load(self.word2vec_model_file)
        else:
            print("Training a new Word2Vec model...")
            self._train_word2vec_model()
            self.save_word2vec_model()


    def save_data(self, data, filename):
        """Saves data to a pickle file."""
        os.makedirs(self.data_dir, exist_ok=True)  # Create data directory if it doesn't exist
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def fetch_binance_news(self):
        """Fetches news articles from Binance."""
        url = "https://www.binance.com/bapi/composite/v3/friendly/pgc/feed/news/list"
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Bnc-Uuid": "REPLACE_WITH_YOUR_BNC_UUID", # Replace with your actual Bnc-Uuid value
            "Clienttype": "web",
            "Content-Type": "application/json",
            "Cookie": "REPLACE_WITH_YOUR_COOKIE_STRING", # Replace with your actual Cookie string
            "Csrftoken": "REPLACE_WITH_YOUR_CSRFTOKEN", # Replace with your actual Csrftoken value
            "Device-Info": "REPLACE_WITH_YOUR_DEVICE_INFO", # Replace with your actual Device-Info value
            "Fvideo-Id": "REPLACE_WITH_YOUR_FVIDEO_ID", # Replace with your actual Fvideo-Id value
            "Fvideo-Token": "REPLACE_WITH_YOUR_FVIDEO_TOKEN", # Replace with your actual Fvideo-Token value
            "Lang": "en",
            "Origin": "https://www.binance.com",
            "Referer": "https://www.binance.com/en/support/faq/c97c9bc9f5894e398610c4a8317b6f2e",
            "Sec-Ch-Ua": "\"Opera GX\";v=\"109\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"Windows\"",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
            "X-Passthrough-Token": "REPLACE_WITH_YOUR_X_PASSTHROUGH_TOKEN", # Replace with your actual X-Passthrough-Token value
            "X-Trace-Id": "REPLACE_WITH_YOUR_X_TRACE_ID", # Replace with your actual X-Trace-Id value
            "X-Ui-Request-Trace": "REPLACE_WITH_YOUR_X_UI_REQUEST_TRACE" # Replace with your actual X-Ui-Request-Trace value
        }
        data = {
            "featured": False,
            "pageIndex": 1,
            "pageSize": 20,
            "strategy": 5,
            "tagId": 16
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            data = response.json()
            if data['code'] == "000000":
                articles = data['data']['vos']
                for article in articles:
                    title = article['title']
                    url = article['webLink']
                    timestamp = article['date']
                    # Convert Unix timestamp to readable date
                    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    sentiment = TextBlob(article['title']).sentiment.polarity
                    self.news_articles.append({
                        'title': title,
                        'url': url,
                        'date': date,
                        'sentiment': sentiment,
                        'content': title  # Add the 'content' key here
                    })
            else:
                print(f"Binance News API Error: {data['code']}")
        else:
            print(f"Binance News API Status: Error ({response.status_code})")
            print(f"Error Message: {response.text}")

    def fetch_coindesk_news(self):
        """Fetches news articles from Coindesk API and analyzes their sentiment."""
        url = 'https://www.coindesk.com/pf/api/v3/content/fetch/please-stop'
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cookie": 'AKA_A2=A; SLG_G_WPT_TO=fr; SLG_GWPT_Show_Hide_tmp=1; SLG_wptGlobTipTmp=1; _pbjs_userid_consent_data=3524755945110770; CookieConsent={stamp:%27qN6TsRPf8KdqtdKYMvZYVgnSNR3HybnxHSu9KT/l18X0sNu7MLp1Bw==%27%2Cnecessary:true%2Cpreferences:false%2Cstatistics:false%2Cmarketing:false%2Cmethod:%27explicit%27%2Cver:8%2Cutc:1718189211579%2Cregion:%27tn%27}; __gads=ID=ac611b876c6d27de:T=1718188180:RT=1718190966:S=ALNI_MZDMh6Ff1EaEEky5cVvocMyUMwSiQ; __gpi=UID=00000d83c0f6bd67:T=1718188180:RT=1718190966:S=ALNI_MaSuXdweNUdetNo36WiRE3HeMIr5w; __eoi=ID=e78b328636dc16c3:T=1718188180:RT=1718190966:S=AA-AfjbJ9YgF-lKfAfixg8De5f_F',
            "Referer": "https://www.coindesk.com/livewire/",
            "Sec-Ch-Ua": '\"Opera GX\";v=\"109\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '\"Windows\"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0"
        }
        params = {
            "query": '{"language":"en","size":20,"page":1,"format":"timeline"}'
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('items', [])
            extracted_articles = []
            for article in articles:
                title = article.get('title', '')
                content = article.get('description', '')
                date = article.get('date', '')
                if date:
                    date = date.split('T')[0]  # Extract only the date part
                # Analyze sentiment
                sentiment = TextBlob(content).sentiment.polarity
                extracted_articles.append({
                    'title': title,
                    'content': content, # Make sure 'content' key exists here
                    'date': date,
                    'sentiment': sentiment
                })
            self.news_articles.extend(extracted_articles)
        else:
            print(f"Coindesk API Error: {response.status_code}")
            print(f"Error Message: {response.text}")

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
    
    def _train_word2vec_model(self):
        """Trains a Word2Vec model using the collected news articles."""
        sentences = [article['content'].split() for article in self.news_articles]
        
        # Create and build the vocabulary
        self.word2vec_model = Word2Vec(vector_size=100, window=5, min_count=5, workers=4)
        self.word2vec_model.build_vocab(sentences)  # Build vocabulary

        # Train the model
        self.word2vec_model.train(sentences, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.epochs)

    def save_word2vec_model(self):
        """Saves the trained Word2Vec model to a file."""
        self.word2vec_model.save(self.word2vec_model_file)

    def _get_article_embedding(self, article_text):
        """Generates a word embedding representation for an article."""
        if self.word2vec_model is None:
            self._train_word2vec_model()  # Train the model if it hasn't been trained yet
        words = article_text.split()
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if vectors:
            return sum(vectors) / len(vectors)
        else:
            return [0] * self.word2vec_model.vector_size

    def cluster_news_articles(self):
        """Clusters news articles based on their content similarity."""
        article_embeddings = [self._get_article_embedding(article['content']) for article in self.news_articles]
        kmeans = KMeans(n_clusters=self.num_clusters)
        clusters = kmeans.fit_predict(article_embeddings)

        # Assign cluster IDs to articles
        for i, article in enumerate(self.news_articles):
            article['cluster_id'] = clusters[i]
    
    def _calculate_sentiment_index(self, news_articles):
        """Calculates the overall sentiment index from the news articles."""
        total_sentiment = 0
        for article in news_articles:
            total_sentiment += article['sentiment']
        if len(news_articles) > 0:
            return total_sentiment / len(news_articles)
        else:
            return 0 
        
    def load_candlesticks(self, filename='completed_candlesticks.csv'):
        """Loads candlestick data from a CSV file, skipping incomplete candles."""
        try:
            candlesticks = []
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check for missing data in 'Low Price' and other columns
                    if row['Low Price'] in (None, '', 'None'):
                        low_price = 0.0  # Default value or another handling strategy
                    else:
                        low_price = float(row['Low Price'])

                    # Apply similar checks for 'Open Price', 'Close Price', and 'High Price'
                    open_price = float(row['Open Price']) if row['Open Price'] not in (None, '', 'None') else 0.0
                    close_price = float(row['Close Price']) if row['Close Price'] not in (None, '', 'None') else 0.0
                    high_price = float(row['High Price']) if row['High Price'] not in (None, '', 'None') else 0.0

                    entry_time = row['Entry Time']
                    exit_time = row['Exit Time']

                    candlesticks.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'open_price': open_price,
                        'close_price': close_price,
                        'high_price': high_price,
                        'low_price': low_price,
                    })
            return candlesticks
        except FileNotFoundError:
            return None
        except Exception as e:
            print("Load candlestick error", e)

    def preprocess_candlestick_data(self, candlesticks, window_size=10):
        """Preprocesses candlestick data for analysis."""
        processed_data = []
        for i in range(window_size, len(candlesticks)):
            # Extract features from the last 'window_size' candles
            features = []
            for j in range(i - window_size, i):
                features.extend([
                    candlesticks[j]['open_price'],
                    candlesticks[j]['high_price'],
                    candlesticks[j]['low_price'],
                    candlesticks[j]['close_price'],
                    # Add more features as needed, such as body length, shadow lengths, etc.
                ])
            # Normalize the features
            features = np.array(features)
            features = (features - np.mean(features)) / np.std(features)

            # Calculate the target (price movement)
            current_price = candlesticks[i]['close_price']
            next_price = candlesticks[i + 1]['close_price'] if i + 1 < len(candlesticks) else current_price
            target = 1 if next_price > current_price else 0

            processed_data.append((features, target))

        return processed_data
    def get_news_articles(self):
        """Returns the fetched news articles."""
        return self.news_articles