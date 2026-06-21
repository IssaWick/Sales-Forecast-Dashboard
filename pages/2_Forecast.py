"""Page 2 - Live Prophet forecast with adjustable horizon."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

from utils import load_data, weekly_sales

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Sales Forecast")
st.write(
    "Prophet retrains on the historical weekly sales and projects future demand. "
    "Adjust the region and horizon below — the model retrains automatically."
)

df = load_data()
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())

c1, c2 = st.columns(2)
region = c1.selectbox("Region", regions)
horizon = c2.slider("Weeks to forecast", min_value=4, max_value=52, value=26, step=2)

ts = weekly_sales(region)


@st.cache_data(show_spinner="Training Prophet model...")
def make_forecast(ts, horizon):
    """Fit Prophet on the series and forecast `horizon` weeks ahead."""
    train_df = pd.DataFrame({"ds": ts.index, "y": ts.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=horizon, freq="W")
    return m.predict(future)


forecast = make_forecast(ts, horizon)
cutoff = ts.index.max()

# ---------- Plot: actuals + forecast + confidence band ----------
fig = go.Figure()
# Upper then lower with fill='tonexty' creates the shaded band between them
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                         mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                         mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(65,105,225,0.15)", name="Confidence interval"))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                         mode="lines", name="Forecast", line=dict(color="royalblue")))
fig.add_trace(go.Scatter(x=ts.index, y=ts.values,
                         mode="lines", name="Actual", line=dict(color="black")))
fig.add_vline(x=cutoff, line_width=1, line_dash="dash", line_color="red")
fig.update_layout(
    title="Historical Sales and Forecast",
    margin=dict(t=50, b=70),
    legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Future-only summary ----------
future_only = forecast[forecast["ds"] > cutoff][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
future_only.columns = ["Week", "Forecast", "Low", "High"]

m1, m2 = st.columns(2)
m1.metric(f"Total forecast (next {horizon} wks)", f"${future_only['Forecast'].sum():,.0f}")
m2.metric("Avg weekly forecast", f"${future_only['Forecast'].mean():,.0f}")

st.subheader("Forecast detail")
st.dataframe(
    future_only.style.format({"Forecast": "${:,.0f}", "Low": "${:,.0f}", "High": "${:,.0f}"}),
    use_container_width=True,
)