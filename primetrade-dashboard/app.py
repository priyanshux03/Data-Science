import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Trader Behavior vs Market Sentiment",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/daily.csv")

daily = load_data()

# =========================
# TITLE
# =========================
st.title("Trader Behavior vs Market Sentiment")
st.write(
    "Interactive dashboard summarizing trader behavior, performance, "
    "and sentiment-driven patterns derived from the analysis."
)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Market Sentiment",
    options=daily["classification"].unique(),
    default=daily["classification"].unique()
)

exposure_filter = st.sidebar.multiselect(
    "Exposure Segment",
    options=daily["exposure_segment"].unique(),
    default=daily["exposure_segment"].unique()
)

filtered = daily[
    (daily["classification"].isin(sentiment_filter)) &
    (daily["exposure_segment"].isin(exposure_filter))
]

# =========================
# OVERVIEW KPIs
# =========================
st.header("1. Overall Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Daily PnL", f"{filtered['daily_pnl'].mean():,.0f}")
col2.metric("Win Rate", f"{filtered['win_rate'].mean():.2f}")
col3.metric("Trades / Day", f"{filtered['trades_count'].mean():.0f}")
col4.metric("Long Ratio", f"{filtered['long_ratio'].mean():.2f}")

# =========================
# BEHAVIOR VS SENTIMENT
# =========================
st.header("2. Trader Behavior vs Market Sentiment")

behavior_table = (
    filtered.groupby("classification")
    .agg(
        avg_trades=("trades_count", "mean"),
        avg_exposure=("avg_exposure", "mean"),
        avg_trade_size=("avg_trade_size", "mean"),
        long_ratio=("long_ratio", "mean")
    )
)

st.dataframe(behavior_table)

# =========================
# PERFORMANCE VS SENTIMENT
# =========================
st.header("3. Performance vs Market Sentiment")

performance_table = (
    filtered.groupby("classification")
    .agg(
        avg_daily_pnl=("daily_pnl", "mean"),
        median_daily_pnl=("daily_pnl", "median"),
        pnl_volatility=("daily_pnl", "std"),
        win_rate=("win_rate", "mean")
    )
)

st.dataframe(performance_table)

# =========================
# VISUAL EVIDENCE
# =========================
st.header("4. Visual Evidence")

# --- PnL Distribution ---
st.subheader("PnL Distribution by Sentiment")
fig, ax = plt.subplots()
filtered.boxplot(column="daily_pnl", by="classification", ax=ax)
ax.set_title("")
plt.suptitle("")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Trade Frequency ---
st.subheader("Trade Frequency by Sentiment")
freq = filtered.groupby("classification")["trades_count"].mean()
st.bar_chart(freq)

# --- Long Bias ---
st.subheader("Long Bias by Sentiment")
long_bias = filtered.groupby("classification")["long_ratio"].mean()
st.bar_chart(long_bias)

# --- Position Size ---
st.subheader("Average Trade Size by Sentiment")
size_chart = filtered.groupby("classification")["avg_trade_size"].mean()
st.bar_chart(size_chart)

# =========================
# SEGMENT ANALYSIS
# =========================
st.header("5. Trader Segments")

tab1, tab2, tab3 = st.tabs(
    ["Exposure (Risk)", "Frequency (Activity)", "Consistency"]
)

with tab1:
    st.subheader("Exposure-Based Segments")
    exposure_table = (
        filtered.groupby(["classification", "exposure_segment"])
        [["daily_pnl", "win_rate", "trades_count", "long_ratio"]]
        .mean()
    )
    st.dataframe(exposure_table)

with tab2:
    st.subheader("Frequency-Based Segments")
    frequency_table = (
        filtered.groupby(["classification", "frequency_segment"])
        [["daily_pnl", "win_rate", "avg_exposure", "long_ratio"]]
        .mean()
    )
    st.dataframe(frequency_table)

with tab3:
    st.subheader("Consistency-Based Segments")
    consistency_table = (
        filtered.groupby(["classification", "consistency_segment"])
        [["daily_pnl", "win_rate", "avg_exposure", "trades_count"]]
        .mean()
    )
    st.dataframe(consistency_table)

# =========================
# BONUS: PREDICTIVE MODEL SUMMARY
# =========================
st.header("6. Bonus: Predictive Model")

st.info(
    "A logistic regression model using sentiment and behavioral features "
    "achieved ~75% accuracy in predicting next-day trader profitability. "
    "This indicates meaningful predictive signal in sentiment-driven behavior."
)

# =========================
# KEY INSIGHTS
# =========================
st.header("Key Insights")

st.markdown("""
- Traders increase exposure, activity, position size, and long bias during Greed and Extreme Greed.
- Despite higher optimism, win rates and average PnL decline in greedy regimes.
- Fear periods consistently show better performance with more disciplined behavior.
- High exposure and high frequency amplify downside risk during optimistic sentiment.
""")
