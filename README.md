Trader Sentiment Analysis and Predictive Modeling
This project analyzes the relationship between trader behavior on the Hyperliquid platform and the broader Bitcoin market sentiment. It uses data science techniques to uncover patterns, develop predictive models, and provide data-driven insights for trading strategies.

dataset:https://drive.google.com/file/d/12h2eEKqo9WtpZ249HaUQNcSzIGZTAnms/view?usp=drive_link,https://drive.google.com/file/d/1dR_Zuf7JNyhLFmQ2KLKmoAQ4NuVUb65N/view?usp=drive_link
live demo:http://localhost:8501/,https://project-sentifi-git-main-rahul-rais-projects-78eaf2ef.vercel.app/
Key Features
NLP-Enhanced Sentiment Analysis: Utilizes the VADER sentiment analyzer to create a nuanced sentiment score, offering a more granular view than traditional classifications.
Predictive PnL Model: A RandomForestRegressor that predicts daily Profit and Loss (PnL) with an R-squared of 33.5%.
Predictive Sentiment Model: A RandomForestClassifier that forecasts the next day's market sentiment with 56.1% accuracy.
Actionable Trading Insights: Identifies key drivers of profitability and sentiment shifts, such as trading volume and recent performance.
Getting Started
Prerequisites
Python 3.x
Pip
Installation
Clone the repository:

git clone https://github.com/your-username/trader-sentiment-analysis.git
cd trader-sentiment-analysis
Install the required libraries:

pip install -r requirements.txt
Running the Application
Place your data files (fear_greed_index.csv and historical_data.csv) in the root directory of the project.

Run the Streamlit application:

streamlit run app.py
Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

Project Overview
This project follows a structured data science workflow:

Data Loading and Preprocessing: The initial step involves loading the raw data, cleaning it to remove invalid entries, and standardizing formats for analysis.
Feature Engineering: New features are created to enhance the predictive power of the models. This includes:
Daily Volatility: The standard deviation of execution prices for each asset.
Rolling Averages: 7-day rolling averages of PnL, volume, and sentiment scores to capture trends.
Lag Features: Past data points are used to predict future outcomes.
Predictive Modeling: Two primary models are developed:
A regression model to predict daily PnL.
A classification model to predict market sentiment.
Hyperparameter Tuning: GridSearchCV is used to optimize the models for the best possible performance.
Key Findings
Top Trader Behavior: Successful traders are highly active, adapt their strategies to market sentiment, and diversify their assets.
PnL Prediction: Recent performance (rolling_7d_pnl) and volatility are the most significant factors in predicting future PnL.
Sentiment Prediction: Trading volume is the strongest indicator of upcoming shifts in market sentiment.
Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.
