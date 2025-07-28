# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- Page Configuration ---
st.set_page_config(
    page_title="Trader Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Title and Introduction ---
st.title("📊 Trader Sentiment Analysis Dashboard")
st.write("""
This application performs a comprehensive analysis of historical trader data and Bitcoin market sentiment.
The goal is to uncover hidden patterns and deliver insights that can drive smarter trading strategies.
""")


# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    try:
        df_sentiment = pd.read_csv('fear_greed_index.csv')
        df_trader = pd.read_csv('historical_data.csv')
        return df_sentiment, df_trader
    except FileNotFoundError:
        st.error("Error: 'fear_greed_index.csv' or 'historical_data.csv' not found. Please make sure they are in the same directory as the app.")
        return None, None

df_sentiment, df_trader = load_data()

if df_sentiment is not None and df_trader is not None:
    # --- Data Preprocessing and Feature Engineering (Cached) ---
    @st.cache_data
    def preprocess_and_feature_engineer(df_sentiment, df_trader):
        # --- 1. Data Cleaning and Preparation ---
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        df_trader['trade_datetime'] = pd.to_datetime(df_trader['Timestamp IST'], format='%d-%m-%Y %H:%M')
        df_trader['trade_date'] = pd.to_datetime(df_trader['trade_datetime'].dt.date)
        df_trader_cleaned = df_trader[(df_trader['Size Tokens'] > 0) & (df_trader['Execution Price'] > 0)].copy()

        # --- 2. NLP Sentiment Analysis ---
        nltk.download('vader_lexicon', quiet=True)
        analyzer = SentimentIntensityAnalyzer()
        df_sentiment['vader_sentiment_score'] = df_sentiment['classification'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

        # --- 3. Feature Engineering ---
        daily_volatility = df_trader_cleaned.groupby(['Coin', 'trade_date'])['Execution Price'].std().reset_index()
        daily_volatility.rename(columns={'trade_date': 'date', 'Execution Price': 'daily_volatility'}, inplace=True)
        daily_volatility['daily_volatility'].fillna(0, inplace=True)

        daily_summary_by_coin = df_trader_cleaned.groupby(['trade_date', 'Coin']).agg(
            daily_pnl=('Closed PnL', 'sum'),
            daily_volume=('Size USD', 'sum'),
            trade_count=('Account', 'count')
        ).reset_index()
        daily_summary_by_coin.rename(columns={'trade_date': 'date'}, inplace=True)

        df_featured = pd.merge(daily_summary_by_coin, df_sentiment, on='date', how='inner')
        df_featured = pd.merge(df_featured, daily_volatility, on=['date', 'Coin'], how='left')
        df_featured['daily_volatility'].fillna(0, inplace=True)

        df_featured = df_featured.sort_values(by=['Coin', 'date'])
        df_featured['rolling_7d_pnl'] = df_featured.groupby('Coin')['daily_pnl'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df_featured['rolling_7d_volume'] = df_featured.groupby('Coin')['daily_volume'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df_featured['lag_1d_pnl'] = df_featured.groupby('Coin')['daily_pnl'].shift(1)
        df_featured['lag_1d_volume'] = df_featured.groupby('Coin')['daily_volume'].shift(1)
        df_featured['lag_1d_sentiment'] = df_featured.groupby('Coin')['vader_sentiment_score'].shift(1)
        df_featured.fillna(0, inplace=True)

        return df_featured, df_trader_cleaned

    df_featured, df_trader_cleaned = preprocess_and_feature_engineer(df_sentiment.copy(), df_trader.copy())


    # --- Model Training (Cached) ---
    @st.cache_resource
    def train_models(df_featured):
        # --- PnL Prediction Model ---
        df_model_pnl = df_featured[df_featured['daily_pnl'] != 0].copy()
        features_pnl = [
            'rolling_7d_pnl', 'rolling_7d_volume', 'lag_1d_pnl', 'lag_1d_volume',
            'lag_1d_sentiment', 'daily_volatility', 'vader_sentiment_score', 'trade_count'
        ]
        target_pnl = 'daily_pnl'
        X_pnl = df_model_pnl[features_pnl]
        y_pnl = df_model_pnl[target_pnl]
        model_pnl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_pnl.fit(X_pnl, y_pnl)

        # --- Sentiment Prediction Model ---
        df_featured['next_day_sentiment'] = df_featured.groupby('Coin')['classification'].shift(-1)
        df_model_sent = df_featured.dropna(subset=['next_day_sentiment']).copy()
        features_sent = [
            'rolling_7d_pnl', 'rolling_7d_volume', 'lag_1d_pnl', 'lag_1d_volume',
            'lag_1d_sentiment', 'daily_volatility', 'vader_sentiment_score', 'trade_count'
        ]
        target_sent = 'next_day_sentiment'
        X_sent = df_model_sent[features_sent]
        y_sent = df_model_sent[target_sent]
        model_sent = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_sent.fit(X_sent, y_sent)

        return model_pnl, model_sent

    model_pnl, model_sent = train_models(df_featured.copy())

    # --- Sidebar for User Input ---
    st.sidebar.header("User Input")
    selected_coin = st.sidebar.selectbox("Select a Coin", df_featured['Coin'].unique())


    # --- Main Dashboard ---
    st.header(f"Analysis for: {selected_coin}")

    # Filter data for the selected coin
    df_coin = df_featured[df_featured['Coin'] == selected_coin]

    # --- PnL and Sentiment Over Time ---
    st.subheader("PnL and Sentiment Over Time")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_coin['date'], df_coin['daily_pnl'], color='tab:blue', label='Daily PnL')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily PnL', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(df_coin['date'], df_coin['vader_sentiment_score'], color='tab:red', linestyle='--', label='Sentiment Score')
    ax2.set_ylabel('Sentiment Score', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    st.pyplot(fig)

    # --- Top Traders ---
    st.sidebar.header("Top Traders")
    top_traders_by_pnl = df_trader_cleaned.groupby('Account')['Closed PnL'].sum().nlargest(5)
    st.sidebar.table(top_traders_by_pnl)


    # --- Predictive Models ---
    st.sidebar.header("Predictive Models")

    # PnL Prediction
    st.sidebar.subheader("Predict Next Day's PnL")
    rolling_pnl_input = st.sidebar.number_input("Rolling 7D PnL", value=df_coin['rolling_7d_pnl'].iloc[-1])
    rolling_volume_input = st.sidebar.number_input("Rolling 7D Volume", value=df_coin['rolling_7d_volume'].iloc[-1])
    lag_pnl_input = st.sidebar.number_input("Lag 1D PnL", value=df_coin['lag_1d_pnl'].iloc[-1])
    lag_volume_input = st.sidebar.number_input("Lag 1D Volume", value=df_coin['lag_1d_volume'].iloc[-1])
    lag_sentiment_input = st.sidebar.number_input("Lag 1D Sentiment", value=df_coin['lag_1d_sentiment'].iloc[-1])
    volatility_input = st.sidebar.number_input("Daily Volatility", value=df_coin['daily_volatility'].iloc[-1])
    sentiment_input = st.sidebar.number_input("Current Sentiment Score", value=df_coin['vader_sentiment_score'].iloc[-1])
    trade_count_input = st.sidebar.number_input("Trade Count", value=df_coin['trade_count'].iloc[-1])

    if st.sidebar.button("Predict PnL"):
        pnl_features = np.array([[
            rolling_pnl_input, rolling_volume_input, lag_pnl_input, lag_volume_input,
            lag_sentiment_input, volatility_input, sentiment_input, trade_count_input
        ]])
        pnl_prediction = model_pnl.predict(pnl_features)
        st.sidebar.success(f"Predicted PnL: ${pnl_prediction[0]:,.2f}")

    # Sentiment Prediction
    st.sidebar.subheader("Predict Next Day's Sentiment")
    if st.sidebar.button("Predict Sentiment"):
        sent_features = np.array([[
            rolling_pnl_input, rolling_volume_input, lag_pnl_input, lag_volume_input,
            lag_sentiment_input, volatility_input, sentiment_input, trade_count_input
        ]])
        sent_prediction = model_sent.predict(sent_features)
        st.sidebar.info(f"Predicted Sentiment: {sent_prediction[0]}")