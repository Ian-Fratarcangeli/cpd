import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta



def generate_gbm(n=500, cp = 250, mu=0.1, sigma=0.25, S_0=100, seed=42):
    dt = 1/252
    features = pd.DataFrame()
    np.random.seed(seed)
    r1 = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=cp)
    r2 = np.random.normal(mu*2 * dt, sigma*2 * np.sqrt(dt), size=n - cp)
    returns = np.concatenate([r1, r2])
    prices = S_0 * np.exp(np.cumsum(returns))
    prices = pd.Series(prices)
    
    features["log_return"] = np.log(prices / prices.shift(1))
    #get 5 and 20 day rolling vols
    features["vol_5d"] = features["log_return"].rolling(window=5).std()
    features["vol_20d"] = features["log_return"].rolling(window=20).std()

    #technical indicators
    features["rsi"] = ta.momentum.RSIIndicator(close=prices).rsi()
    features["macd"] = ta.trend.MACD(close=prices).macd_diff()
    
    #ma
    features["sma_5"] = prices.rolling(window=5).mean()
    features["sma_20"] = prices.rolling(window=20).mean()
    #ema
    features["ema_12"] = prices.ewm(span=12).mean()

    #derivative
    features["accel"] = features["log_return"].diff()

    features = features.dropna()

    #scale the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, returns, prices
