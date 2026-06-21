"""Shared helpers used by every page of the dashboard."""
import pandas as pd
import streamlit as st

DATA_PATH = "data/Superstore.csv"


@st.cache_data
def load_data():
    """Load and clean the raw Superstore data once, then reuse from cache."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])
    return df


@st.cache_data
def weekly_sales(region="All", category="All"):
    """Return a weekly sales series, optionally filtered by region/category."""
    df = load_data()
    if region != "All":
        df = df[df["Region"] == region]
    if category != "All":
        df = df[df["Category"] == category]
    ts = df.set_index("Order Date")["Sales"].resample("W").sum().sort_index()
    return ts