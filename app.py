import streamlit as st
from model import load_data, train_model, predict_and_plot
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting ML", page_icon="📈")
st.title("📈 Sales Forecasting using XGBoost")

data = load_data("data/SuperMarket_Analysis.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

model, X_test, y_test = train_model(data)

preds = model.predict(X_test)

st.subheader("Actual vs Predicted Sales")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label="Actual")
ax.plot(preds, label="Predicted")
ax.legend()
ax.set_title("Sales Prediction")
st.pyplot(fig)