import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
from data import compile_features
from optimize import generate_gbm 
from cusum_like import online_cusum_like
from online_cpd import online_kernel_cusum
from sklearn.metrics import pairwise_distances
from pandas.tseries.offsets import BDay

# === Streamlit App ===

st.title("Online CUSUM Change Point Detection")

st.header("Introduction")
st.write("""Many times, when looking at financial data, we ask ourselves, "Is this volatility just noise or a shift in the market?"
          Change point detection (CPD) attempts to define points within time-series data where a true distribution change has occurred. 
         There are many different methods for CPD, and here we are going to test some different ways we can apply the Cumulative Sum Control Chart (CUSUM) technique. CUSUM 
         works to detect change points by accumulating evidence of distributional shift through a comparison function over sliding windows. A reference window is generated
         and compared to the new point at current time t (since this is online), producing some score of differentiation. When the scores sum above a 
         certain threshold, we can define that point in time t as a change point, and the process restarts. We are going to be looking at CUSUM applied with likelihood and kernel methodology.
         The symbol sidebar can be used input an equity of your choice and the date range to define the time the CUSUM will monitor. The window size and alpha, 
         which will be explained more in depth later, can also be adjusted.
""")
symbol = st.sidebar.text_input("Symbol", "SPY")
today = datetime.today().strftime("%Y-%m-%d")
start_date = st.text_input("Start Date of Period (YYYY-MM-DD)", value="2023-01-01")
end_date = st.text_input("End Date of Period (YYYY-MM-DD)", value=today)
#validate the date inputs
valid_date = None
if start_date:
    try:
        valid_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        if valid_date > datetime.today().date():
            st.error("Quote dates cannot be in the future. Please try again.")
        else: st.success(f"Valid date: {valid_date}")
    except ValueError:
        st.error("Invalid date format. Please enter date as YYYY-MM-DD.")
else: st.error("Please input a date for the start date.")
valid_date = None
if end_date:
    try:
        valid_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if valid_date > datetime.today().date():
            st.error("Quote dates cannot be in the future. Please try again.")
        elif datetime.strptime(start_date, "%Y-%m-%d").date() > valid_date:
            st.error("End date cannot be before the start date. Please try again.")
        else: st.success(f"Valid date: {valid_date}")
    except ValueError:
        st.error("Invalid date format. Please enter date as YYYY-MM-DD.")
else: st.error("Please input a date for the end date.")

# Parameters
window = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=50)
alpha = st.sidebar.slider("Alpha (EWMA)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

# Load & preprocess data
#need to give time for the technical indicators to have enough data to initialize
data_delay = 35
better_date = (datetime.strptime(start_date, "%Y-%m-%d") - BDay(window+data_delay)).strftime("%Y-%m-%d")

features_unscaled = compile_features(symbol, better_date, end_date)
features_unscaled = features_unscaled[features_unscaled.index >= (datetime.strptime(start_date, "%Y-%m-%d")- BDay(window))]

prices = features_unscaled["price"][window:]
log_returns = features_unscaled["log_return"][window:]
dates = features_unscaled.index[window:]

features_unscaled = features_unscaled.drop(columns=["price", "log_return"])
columns = features_unscaled.columns
#prepare the features and dates for cpd detection and plotting
scaler = StandardScaler()
features = scaler.fit_transform(features_unscaled)

st.header("Data")
st.write("""Pictured below is a sample of the feature data we will be using for CPD over the given time period.
          The features are calculated from yfinance equity price history and are given a standard transform. As you can see, most of the features have some relation to volatility,
         so the output change points provide timestamps of rapid changes in volatility that aren't characteristic with prior data.""")

feature_descriptions = {
    "vol_5d": "5-day rolling standard deviation of log returns",
    "vol_20d": "20-day rolling standard deviation of log returns",
    "volume_z": "Z-score of trading volume based on a 20-day rolling window",
    "rsi": "Relative Strength Index (RSI), seen as a momentum indicator",
    "macd": "Moving Average Convergence/Divergence (MACD), also a momentum indicator",
    "atr": "Average True Range (ATR), a measure of market volatility",
    "accel": "First derivative of log returns"
}

# Convert to a DataFrame for display
desc_df = pd.DataFrame(list(feature_descriptions.items()), columns=["Feature", "Description"])

# Display with Streamlit
st.subheader("Feature Descriptions")
st.dataframe(desc_df, use_container_width=True)
###THIS IS THE LIKELIHOOD BASED CUSUM
st.header("Likelihood based CUSUM")
st.write("""A log likelihood ratio function is used to compute a score that represents the likelihood that a new point doesn't belong in the prior distribution 
         and instead a new distribution, given a shared covariance. The log likelihood ratio is a parametric method; to calculate likelihood, a covariance,
         a mean for the prior distribution, and a mean for the new distribution are required. The covariance is generated from the prior distribution, which is the 
         data seen in the reference window. We can then initialize the new distribution mean as the prior distribution mean and update it over time as we see more data.
         The new distribution mean is updated using exponential weighted moving average (EWMA) with an alpha hyperparameter that can be adjusted.
         A potential issue here is that the assumptions we made for covariance and means may not be true. It is very possible that the true mean for this distribution 
         is not what we have estimated from our prior distribution window, resulting in inaccurate likelihood scores and consequently false change points.
""")
st.write("""As the likelihood scores sum, the CUSUM algorithm compares it to a threshold to see if there is a change point at that time. 
         As a result, the threshold is extremely significant and needs to be designated in order to best avoid false positives and true negatives. 
         To do this, a geometric brownian motion with a distributional shift has been generated and the relevant features calculated to fit the data we want. 
         This data is sent to the likelihood CUSUM and scores are produced to give a better idea of how the method reacts to financial equity price data (GBM is considered a good simulation of financial data).
         The 99th percentile of scores is then used as the threshold score.
""")
cp = 250
n = 500
optimize_features, _, _ = generate_gbm(n=n, cp=cp)
_, optimize_scores, _ = online_cusum_like(optimize_features, window_size=window)
threshold = np.percentile(optimize_scores, 99)
#this is based on the mean
drift = 1.10

st.write(f"Optimal threshold set at 99th percentile: {threshold:.2f}")

alarms, scores, _ = online_cusum_like(features, window_size=window, alpha=alpha, h=threshold, drift=drift)
st.header("Results")
st.write("Below the CUSUM scores using log likelihood ratio are compared to the log returns and equity price, with the red dotted vertical lines representing the change points.")
# === Plot 1: Log returns with CUSUM scores ===

fig1, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Log Returns', color=color)
ax1.plot(dates, log_returns, color=color, label='Log Returns')
ax1.tick_params(axis='y', labelcolor=color)

for t in alarms:
    if t < len(dates):
        ax1.axvline(dates[t], color='red', linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('CUSUM Score', color=color)
ax2.plot(dates, scores, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
fig1.tight_layout()

st.pyplot(fig1)

# === Plot 2: Asset price with alarms ===
fig2, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_xlabel('Date')
ax3.set_ylabel('Price', color='tab:orange')
ax3.plot(dates, prices, color='tab:orange', label='Price')
ax3.tick_params(axis='y', labelcolor='tab:orange')

for t in alarms:
    if t < len(dates):
        ax3.axvline(dates[t], color='red', linestyle='--', alpha=0.6)

fig2.tight_layout()
st.pyplot(fig2)

# === Alarm report ===
if alarms:
    with st.expander("View detected change point dates"):
        alarm_dates = [str(dates[t].date()) for t in alarms if t < len(dates)]
        st.write(alarm_dates)
else:
    st.subheader("No change points detected.")

###THIS IS THE KERNEL BASED CUSUM
st.header("Kernel based CUSUM")
st.write("""The parametric method above forced us to make assumptions on the distribution. Kernel based methods are non-parametric, meaning conclusions
         can be drawn without having to estimate any distributional parameters. Maximum Mean Disrepancy (MMD) is the kernel-based method chosen, which takes the reference window and new point
         and maps them both to Reproducing Kernel Hilbert Space. Essentially, MMD is measuring the distance between the reference probability distribution and the new point.
         This method requires a parameter for kernel bandwith (gamma) describing how much each point matters, which is created using the median heuristic method derived from euclidean distances.
""")
dists = pairwise_distances(features, metric='euclidean')
# Get the upper triangle, excluding the diagonal
dists = dists[np.triu_indices_from(dists, k=1)]
median_sq_dist = np.median(dists ** 2)
alpha = 1.2
gamma = (1.0 / (2 * median_sq_dist)) * alpha
st.write(f"Optimal gamma using median heuristic method: {gamma:.2f}")
#pulled from mean mmd2 scores in a reference
nu = 0.495
h = np.log(1/gamma)

st.write(f"Optimal threshold calculated from logarithm of inverse gamma: {h:.2f}")
alarms_kernel, scores_kernel, _ = online_kernel_cusum(features, window_size=window, gamma=gamma, nu=nu, h=h)
st.header("Results")
# === Plot 1: Log returns with CUSUM scores ===
fig1, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Log Returns', color=color)
ax1.plot(dates, log_returns, color=color, label='Log Returns')
ax1.tick_params(axis='y', labelcolor=color)

for t in alarms_kernel:
    if t < len(dates):
        ax1.axvline(dates[t], color='red', linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('CUSUM Score', color=color)
ax2.plot(dates, scores_kernel, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
fig1.tight_layout()

st.pyplot(fig1)

# === Plot 2: Asset price with alarms ===
fig2, ax3 = plt.subplots(figsize=(12, 6))
ax3.set_xlabel('Date')
ax3.set_ylabel('Price', color='tab:orange')
ax3.plot(dates, prices, color='tab:orange', label='Price')
ax3.tick_params(axis='y', labelcolor='tab:orange')

for t in alarms_kernel:
    if t < len(dates):
        ax3.axvline(dates[t], color='red', linestyle='--', alpha=0.6)

fig2.tight_layout()
st.pyplot(fig2)

# === Alarm report ===
if alarms_kernel:
    with st.expander("View detected change point dates"):
        alarm_dates = [str(dates[t].date()) for t in alarms_kernel if t < len(dates)]
        st.write(alarm_dates)
else:
    st.subheader("No change points detected.")

st.header("Discussion")
st.write("""I think the kernel method outperformed the likelihood method because it was able to better detect big market changes from my testing (COVID, Tariffs, etc.)
         without producing as many false positives as the likelihood method. My assumption is that the covariance estimated for the parametric method was not a very good
         representation of the market distribution; it probably underestimated the market variance. Because of this, new data was seen as less likely than it should have been, 
         resulting in the occurrence of more change points. This project certainly demonstrates the power of non-parametric methods on financial time-series. Since both these
         methods are online, it is possible that their change points could serve as legitimate trading signals.""")