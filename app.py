import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import load_data, train_model, predict_future

st.set_page_config(page_title="Supermarket Sales Forecasting", page_icon="🛒", layout="wide")

st.title("🛒 Supermarket Sales Forecasting")
st.markdown("Predict future sales by product line using XGBoost ML model.")

# Load and train
data = load_data("data/SuperMarket_Analysis.csv")
models = train_model(data)

# Sidebar inputs
st.sidebar.header("🔍 Forecast Settings")
product = st.sidebar.selectbox("Select Product Line", sorted(data["Product line"].unique()))
future_days = st.sidebar.slider("Forecast how many days ahead?", 7, 90, 30)
start_date = st.sidebar.date_input("Start forecast from", value=pd.Timestamp("2019-04-01"))

# Predict
forecast_df = predict_future(models[product], product, start_date, future_days)

# KPI metrics
st.subheader(f"📊 Forecast Summary for: **{product}**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Forecast Days", future_days)
col2.metric("💰 Avg Predicted Sales", f"${forecast_df['Predicted Sales'].mean():.2f}")
col3.metric("📈 Max Predicted Sales", f"${forecast_df['Predicted Sales'].max():.2f}")
col4.metric("📉 Min Predicted Sales", f"${forecast_df['Predicted Sales'].min():.2f}")

# Line chart
st.subheader("📈 Predicted Sales Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(forecast_df["Date"], forecast_df["Predicted Sales"], color="royalblue", linewidth=2, marker="o", markersize=3)
ax1.fill_between(forecast_df["Date"], forecast_df["Predicted Sales"], alpha=0.2, color="royalblue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Predicted Sales ($)")
ax1.set_title(f"Sales Forecast - {product}")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Bar chart
st.subheader("📊 Weekly Sales Forecast (Bar Chart)")
forecast_df["Week"] = forecast_df["Date"].dt.strftime("Week %U")
weekly = forecast_df.groupby("Week")["Predicted Sales"].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(weekly["Week"], weekly["Predicted Sales"], color="coral", edgecolor="black")
ax2.set_xlabel("Week")
ax2.set_ylabel("Total Predicted Sales ($)")
ax2.set_title("Weekly Sales Forecast")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Product comparison
st.subheader("🏆 Product Line Comparison (Avg Forecast)")
all_preds = {}
for p in sorted(data["Product line"].unique()):
    df_p = predict_future(models[p], p, start_date, future_days)
    all_preds[p] = df_p["Predicted Sales"].mean()

comp_df = pd.DataFrame(list(all_preds.items()), columns=["Product Line", "Avg Predicted Sales"])
comp_df = comp_df.sort_values("Avg Predicted Sales", ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 5))
colors = ["green" if p == product else "steelblue" for p in comp_df["Product Line"]]
bars = ax3.barh(comp_df["Product Line"], comp_df["Avg Predicted Sales"], color=colors, edgecolor="black")
ax3.set_xlabel("Avg Predicted Sales ($)")
ax3.set_title("All Product Lines - Avg Predicted Sales")
for bar, val in zip(bars, comp_df["Avg Predicted Sales"]):
    ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"${val:.1f}", va="center")
st.pyplot(fig3)

# Raw historical data
st.subheader("📋 Historical Sales Data")
hist = data[data["Product line"] == product][["Date", "Sales", "Quantity", "Rating"]].sort_values("Date")
st.dataframe(hist.tail(20), use_container_width=True)

# Forecast table
st.subheader("📅 Forecast Table")
st.dataframe(forecast_df[["Date", "Predicted Sales"]].assign(**{"Predicted Sales": forecast_df["Predicted Sales"].round(2)}), use_container_width=True)

# Insights
st.subheader("💡 Insights & Recommendations")
best_product = comp_df.iloc[-1]["Product Line"]
worst_product = comp_df.iloc[0]["Product Line"]
st.success(f"✅ **Best performing product:** {best_product} — highest predicted sales")
st.warning(f"⚠️ **Needs attention:** {worst_product} — lowest predicted sales")
st.info(f"📌 **Selected product ({product})** avg forecast: ${all_preds[product]:.2f}/day over next {future_days} days")