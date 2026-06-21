"""Page 3 - Model performance and methodology."""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")
st.title("🏆 Model Performance")
st.write(
    "How the three models scored on the held-out test set — the most recent 26 weeks, "
    "which none of them saw during training."
)

results = pd.read_csv("models/model_results.csv")

# ---------- Winner callout ----------
best = results.sort_values("MAPE").iloc[0]
st.success(f"Best model: **{best['model']}** — lowest error at {best['MAPE']:.1f}% MAPE.")

# ---------- Scores table ----------
st.subheader("Scores")
st.dataframe(
    results.set_index("model").style.format(
        {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE": "{:.1f}%"}
    ),
    use_container_width=True,
)

# ---------- Comparison chart ----------
st.subheader("Compare")
metric = st.radio("Metric", ["MAPE", "MAE", "RMSE"], horizontal=True)
chart_df = results.sort_values(metric)
fig = px.bar(chart_df, x="model", y=metric,
             text_auto=".1f" if metric == "MAPE" else ".0f")
fig.update_layout(margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)
st.caption("Lower is better for all three metrics.")

# ---------- Methodology ----------
st.subheader("How to read this")
st.markdown(
    """
- **Chronological split:** models trained on the earliest weeks and tested on the
  most recent 26 — no shuffling, mirroring how forecasting works in reality.
- **MAPE** is the average percentage the forecast is off by. It's the headline metric here.
- **Both real models beat the seasonal-naive baseline**, which is the proof that the
  modeling added real value over a no-effort guess.
- **Prophet edged out XGBoost** because this series is fairly short and strongly seasonal —
  the conditions a dedicated trend/seasonality model handles best, while XGBoost's lag
  features have less history to learn from.
    """
)