\# Trader Behavior vs Market Sentiment



\## Objective

Analyze how market sentiment (Fear/Greed) impacts trader behavior and performance on Hyperliquid, and derive actionable trading insights.



---



\## Datasets

\- \*\*Bitcoin Fear \& Greed Index\*\* – daily sentiment classification  

\- \*\*Hyperliquid Historical Trader Data\*\* – trade-level execution data



---



\## Methodology

1\. Converted trade timestamps to daily granularity

2\. Aligned trader activity with daily sentiment

3\. Engineered trader-level metrics:

&nbsp;  - Daily PnL

&nbsp;  - Win rate

&nbsp;  - Trade frequency

&nbsp;  - Average trade size

&nbsp;  - Directional bias (long ratio)

&nbsp;  - Exposure (start position as leverage proxy)

4\. Segmented traders by:

&nbsp;  - Exposure (risk)

&nbsp;  - Trade frequency (activity)

&nbsp;  - Performance consistency

5\. Analyzed behavior and performance across sentiment regimes

6\. Built a lightweight Streamlit dashboard for exploration



---



\## Key Insights

\- Traders increase exposure, trade size, and long bias during Greed and Extreme Greed.

\- Despite higher optimism, win rates and average PnL decline during greedy regimes.

\- Fear periods consistently show better performance with more disciplined behavior.

\- High exposure and high frequency amplify downside risk during optimistic sentiment.



---



\## Actionable Strategies

\- \*\*Risk Control:\*\* Reduce exposure during Greed and Extreme Greed, especially for high-risk traders.

\- \*\*Opportunity Selection:\*\* Favor selective, disciplined trading during Fear periods where risk-adjusted returns are higher.



---



\## Bonus

\- A logistic regression model achieved ~75% accuracy in predicting next-day trader profitability using sentiment and behavior features.

\- A Streamlit dashboard was built to visualize and explore results interactively.



---



\## How to Run

```bash

pip install -r requirements.txt

python -m streamlit run app.py



