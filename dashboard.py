"""
Streamlit Dashboard
--------------------
Run with: streamlit run dashboard.py

Visualizes the multi-agent Decision Intelligence Platform output:
demand forecast, risk signals, and the final explainable recommendation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from orchestrator import DecisionIntelligenceOrchestrator
from data.mock_data import generate_dataset

st.set_page_config(page_title="Decision Intelligence Platform", layout="wide")

st.title("🧠 AI-Powered Decision Intelligence Platform")
st.caption("Multi-agent system | Google Cloud (BigQuery/Vertex AI) + NVIDIA-accelerated forecasting")

with st.sidebar:
    st.header("⚙️ Controls")
    n_days = st.slider("Historical data window (days)", 60, 365, 180)
    horizon = st.slider("Forecast horizon (days)", 3, 14, 7)
    run_btn = st.button("▶ Run Multi-Agent Pipeline", type="primary")

if "output" not in st.session_state or run_btn:
    df = generate_dataset(n_days=n_days)
    orchestrator = DecisionIntelligenceOrchestrator()
    orchestrator.forecasting_agent.horizon = horizon
    st.session_state.output = orchestrator.run_pipeline(df)
    st.session_state.df = df

output = st.session_state.output
df = st.session_state.df

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Forecast Trend", output["forecast"]["trend"].upper())
with col2:
    st.metric("Risk Level", output["risk"]["risk_level"])
with col3:
    st.metric("Decision", output["recommendation"]["decision"].replace("_", " "))

st.divider()

# ---- Demand chart with forecast overlay ----
st.subheader("📈 Demand Forecast — ForecastingAgent")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["demand"], name="Historical Demand", line=dict(color="#4C8BF5")))
fig.add_trace(go.Scatter(
    x=pd.to_datetime(output["forecast"]["forecast_dates"]),
    y=output["forecast"]["forecast_values"],
    name="Forecast",
    line=dict(color="#F5A623", dash="dash"),
))
fig.update_layout(height=400, margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Model confidence: {output['forecast']['confidence']*100:.0f}%")

# ---- Risk / anomaly chart ----
st.subheader("🚨 Supply Chain Risk — RiskAgent")
anomaly_dates = pd.to_datetime(output["risk"]["anomaly_dates"])
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["date"], y=df["inventory_level"], name="Inventory Level", line=dict(color="#2ECC71")))
if len(anomaly_dates) > 0:
    anomaly_vals = df[df["date"].isin(anomaly_dates)]["inventory_level"]
    fig2.add_trace(go.Scatter(
        x=anomaly_dates, y=anomaly_vals, mode="markers", name="Anomaly",
        marker=dict(color="red", size=12, symbol="x"),
    ))
fig2.update_layout(height=350, margin=dict(t=20))
st.plotly_chart(fig2, use_container_width=True)

if output["risk"]["low_stock_alert"]:
    st.error(f"⚠️ Low stock alert — current inventory: {output['risk']['current_inventory']} units")

# ---- Final recommendation ----
st.subheader("✅ Recommendation — RecommendationAgent")
rec = output["recommendation"]
st.success(f"**Decision: {rec['decision'].replace('_', ' ')}**  (confidence: {rec['confidence']*100:.0f}%)")
st.write(f"**Explanation:** {rec['explanation']}")
st.caption(f"Reasoning mode: {rec['reasoning_mode']} — swap to `llm` by setting ANTHROPIC_API_KEY")

with st.expander("🔍 Raw agent outputs (for judges / debugging)"):
    st.json(output["forecast"])
    st.json(output["risk"])
    st.json(output["recommendation"])
