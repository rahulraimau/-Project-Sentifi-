# 📈 Trader Sentiment Analysis

This project explores the relationship between trader performance and Bitcoin market sentiment using two datasets:
1. **Hyperliquid Historical Trader Data** – includes execution details like account, size, symbol, leverage, and closedPnL.
2. **Bitcoin Fear & Greed Index** – daily sentiment classification of the market (Fear or Greed).

## 🧠 Objective
To uncover patterns and correlations between market sentiment and trader outcomes to drive smarter trading strategies.

## 📊 Key Analyses
- Exploratory Data Analysis (EDA) on trader behavior
- Merging sentiment with trading records
- DBSCAN clustering of trading behavior
- Time-series forecasting of market sentiment
- Visualization and actionable insights

## 📁 Files Included
- `Trader_Sentiment_Analysis_Final.ipynb` – Complete notebook with code and results
- `Trader_Sentiment_Report.pdf` – Visual + textual insight report
- `fear_greed_index.csv` – Market sentiment data
- `historical_data.csv` – Trader performance data

## 🛠️ Requirements

Install via pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly statsmodels openpyxl
