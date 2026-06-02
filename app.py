import streamlit as st
import os
from model import load_data, train_model, predict_and_plot

st.set_page_config(page_title="Sales Forecasting ML", page_icon="📈")

st.title("📈 Sales Forecasting using XGBoost")

data = load_data("data/SuperMarket_Analysis.csv")

