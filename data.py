import yfinance as yf
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import os
import seaborn as sns

def get_historical_returns(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get historical log returns data over previous trading days for a given symbol and date period using yfinance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        start_date (str): The date to begin historical returns, (YYYY-MM-DD) form
        end_date (str): The date to end historical returns, (YYYY-MM-DD) form
        
    Returns:
        pd.DataFrame: DataFrame with historical returns
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)

        #calculate log returns
        returns = np.log(df['Close'] / df['Close'].shift(1))
        returns = returns.dropna()

        return returns
    
    except Exception as e:
        print(f"Issue downloading data for {symbol} between {start_date} and {end_date}: {e}")
        return returns 


def compile_features(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Compile a dataframe of important features for change-point detection
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        start_date (str): The date to begin historical returns, (YYYY-MM-DD) form
        end_date (str): The date to end historical returns, (YYYY-MM-DD) form
        
    Returns:
        pd.DataFrame: DataFrame of core, technical, and derivative features
    """
    prices = yf.download(symbol, start=start_date, end=end_date)

    features = pd.DataFrame()
    features["price"] = prices["Close"]
    features["log_return"] = returns = np.log(prices['Close'] / prices['Close'].shift(1))
    
   
    #get 5 and 20 day rolling vols
    features["vol_5d"] = features["log_return"].rolling(window=5).std()
    features["vol_20d"] = features["log_return"].rolling(window=20).std()

    #use the z-score for trading volume based on 20 day rolling window
    features["volume_z"] = (prices["Volume"] - prices["Volume"].rolling(20).mean()) / prices["Volume"].rolling(20).std()

    #technical indicators
    features["rsi"] = ta.momentum.RSIIndicator(close=prices["Close"].squeeze()).rsi()
    features["macd"] = ta.trend.MACD(close=prices["Close"].squeeze()).macd_diff()
    features["atr"] = ta.volatility.AverageTrueRange(high=prices["High"].squeeze(), low=prices["Low"].squeeze(), close=prices["Close"].squeeze()).average_true_range()

    #derivative
    features["accel"] = features["log_return"].diff()
    features = features.dropna()
    return features

def plot_feature_distributions_and_timeseries(features, save_dir=None):
    for col in features.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        # Time series
        axes[0].plot(features.index, features[col], label=col)
        axes[0].set_title(f"Time Series: {col}")
        axes[0].legend()
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        # Distribution
        sns.histplot(features[col], kde=True, ax=axes[1], bins=40)
        axes[1].set_title(f"Distribution: {col}")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{col}_plot.png"))
        plt.tight_layout()
        plt.show()
