"""Page 1 - Interactive sales overview with filters."""
import streamlit as st
import plotly.express as px

from utils import load_data, weekly_sales

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Sales Overview")

df = load_data()

# ---------- Filters ----------
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())

f1, f2 = st.columns(2)
region = f1.selectbox("Region", regions)
category = f2.selectbox("Category", categories)

# Apply the chosen filters to a working copy of the data
fdf = df.copy()
if region != "All":
    fdf = fdf[fdf["Region"] == region]
if category != "All":
    fdf = fdf[fdf["Category"] == category]

# ---------- KPIs (recalculate with every filter change) ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Sales", f"${fdf['Sales'].sum():,.0f}")
k2.metric("Profit", f"${fdf['Profit'].sum():,.0f}")
k3.metric("Orders", f"{fdf['Order ID'].nunique():,}")
aov = fdf["Sales"].sum() / max(fdf["Order ID"].nunique(), 1)
k4.metric("Avg Order Value", f"${aov:,.0f}")

st.divider()

# ---------- Weekly trend (filtered) ----------
st.subheader("Weekly Sales Trend")
ts = weekly_sales(region, category)
ts_df = ts.reset_index()
ts_df.columns = ["Week", "Sales"]
fig_trend = px.line(ts_df, x="Week", y="Sales")
fig_trend.update_layout(margin=dict(t=20, b=20))
st.plotly_chart(fig_trend, use_container_width=True)

# ---------- Breakdowns side by side ----------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Top Sub-Categories")
    sub = (fdf.groupby("Sub-Category")["Sales"].sum()
           .sort_values(ascending=True).tail(10).reset_index())
    fig_sub = px.bar(sub, x="Sales", y="Sub-Category", orientation="h")
    fig_sub.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_sub, use_container_width=True)

with col_b:
    st.subheader("Sales by Segment")
    seg = fdf.groupby("Segment")["Sales"].sum().reset_index()
    fig_seg = px.pie(seg, names="Segment", values="Sales", hole=0.4)
    fig_seg.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_seg, use_container_width=True)

# ---------- Monthly seasonality ----------
st.subheader("Average Sales by Month")
m_ts = fdf.set_index("Order Date")["Sales"].resample("MS").sum()
monthly = m_ts.groupby(m_ts.index.month).mean().reset_index()
monthly.columns = ["Month", "Avg Sales"]
fig_month = px.bar(monthly, x="Month", y="Avg Sales")
fig_month.update_layout(margin=dict(t=20, b=20), xaxis=dict(dtick=1))
st.plotly_chart(fig_month, use_container_width=True)
st.caption("Month 11 (November) usually stands out — the holiday seasonality you spotted in EDA.")