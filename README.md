# 📈 Change Point Detection

This project implements an **online change point detection (CPD)** system for financial time series using both parametric and non-parametric methods. The app runs via a Streamlit dashboard.

---

## 🔍 Overview

The app performs the following:

- Detects change points in financial time series (CUSUM)
- Uses real-time options data via `yfinance`
- Visualizes market data, indicators, and CPD results interactively

---

## 🛠 Features

- **Kernel based CUSUM** (non-parametric)
- **Likelihood-ratio CUSUM** (parametric)
- **Streamlit UI** for interactive analysis
