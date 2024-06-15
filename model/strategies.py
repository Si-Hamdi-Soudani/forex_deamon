from trading_strategy import TradeStrategy, TradeSignal

class SentimentStrategy(TradeStrategy):
    def __init__(self, sentiment_threshold=0.2):
        super().__init__("Sentiment-Based Strategy")
        self.sentiment_threshold = sentiment_threshold

    def should_generate_signal(self, trading_model):
        sentiment_score = trading_model.calculate_sentiment_score()
        return abs(sentiment_score) >= self.sentiment_threshold  # Trade on strong sentiment

    def generate_signal(self, latest_price):
        if trading_model.calculate_sentiment_score() > 0:
            return TradeSignal(self.name, "buy")
        else:
            return TradeSignal(self.name, "sell") 