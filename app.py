"""Landing page for the Sales Forecasting Dashboard."""
import streamlit as st
import plotly.express as px

from utils import load_data, weekly_sales

st.set_page_config(page_title="Sales Forecast Dashboard", page_icon="📈", layout="wide")

df = load_data()

st.title("📈 Sales Forecasting Dashboard")
st.write(
    "An end-to-end time series project on Superstore sales. Explore historical "
    "performance, forecast future demand with Prophet, and compare model accuracy. "
    "Use the pages in the sidebar to navigate."
)

# ---------- Key metrics ----------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
n_orders = df["Order ID"].nunique()
date_min = df["Order Date"].min().date()
date_max = df["Order Date"].max().date()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${total_sales:,.0f}")
c2.metric("Total Profit", f"${total_profit:,.0f}")
c3.metric("Unique Orders", f"{n_orders:,}")
c4.metric("Data Span", f"{date_min.year}-{date_max.year}")

st.divider()

# ---------- Overall weekly trend ----------
st.subheader("Weekly Sales Over Time")
ts = weekly_sales()
ts_df = ts.reset_index()
ts_df.columns = ["Week", "Sales"]
fig = px.line(ts_df, x="Week", y="Sales")
fig.update_layout(margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

st.info("Open the Overview, Forecast, and Model Performance pages from the sidebar.")