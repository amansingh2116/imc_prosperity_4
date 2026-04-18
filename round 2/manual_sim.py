import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Invest & Expand Optimizer", layout="centered")

st.title("📈 Invest & Expand: Market Maker Optimizer")
st.markdown("""
This simulator mathematically finds the perfect **Research** and **Scale** distribution for any given **Speed** budget you choose, based on the IMC Prosperity rules.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Your Strategy")
st.sidebar.markdown("Predict the competition and set your Speed multiplier.")

z = st.sidebar.slider(
    "Speed Investment % (z)", 
    min_value=0, max_value=100, value=34, step=1,
    help="How much of your 100% budget are you dedicating to latency?"
)

m = st.sidebar.slider(
    "Expected Rank Multiplier (M)", 
    min_value=0.1, max_value=0.9, value=0.7, step=0.001,
    help="0.9 is 1st place, 0.1 is last place."
)

# --- Optimization Logic ---
# Remaining budget for Research and Scale
B = 100 - z

# Brute-force integer optimization to maximize (B - x) * ln(1 + x)
best_x = 0
best_val = -1

for x in range(B + 1):
    # Scale = B - x
    val = (B - x) * np.log(1 + x)
    if val > best_val:
        best_val = val
        best_x = x

# Optimal outputs
y = B - best_x

# --- PnL Calculations ---
# Research Formula: 200,000 * ln(1+x) / ln(101)
R_x = 200_000 * np.log(1 + best_x) / np.log(101)

# Scale Formula: 0.07 * y
S_y = 0.07 * y

# Final PnL
total_cost = 50_000  # We always use 100% of budget (100 * 500)
gross_pnl = R_x * S_y * m
net_pnl = gross_pnl - total_cost

# --- Display Results ---
st.header("1. Optimal Allocation")
st.markdown("To maximize your PnL with a **{}%** Speed budget, you must split the remaining **{}%** like this:".format(z, B))

col1, col2, col3 = st.columns(3)
col1.metric("🔬 Optimal Research", f"{best_x}%")
col2.metric("⚖️ Optimal Scale", f"{y}%")
col3.metric("⚡ Chosen Speed", f"{z}%")

st.header("2. Projected Profit")
col4, col5 = st.columns(2)
col4.metric("Gross PnL", f"{gross_pnl:,.0f} XIRECs")

# Highlight Net PnL based on profitability
if net_pnl > 0:
    col5.metric("Net PnL (Profit)", f"💰 {net_pnl:,.0f} XIRECs: R={R_x}, S={S_y}, M={m}")
else:
    col5.metric("Net PnL (Loss)", f"📉 {net_pnl:,.0f} XIRECs: R={R_x}, S={S_y}, M={m}")

# --- Visualization ---
st.header("3. Budget Distribution")

# Create a donut chart using Plotly
df = pd.DataFrame({
    "Pillar": ["Research", "Scale", "Speed"],
    "Allocation": [best_x, y, z]
})

fig = px.pie(
    df, 
    values='Allocation', 
    names='Pillar', 
    hole=0.4,
    color='Pillar',
    color_discrete_map={
        "Research": "#2E86AB",
        "Scale": "#F24236",
        "Speed": "#F5F749"
    }
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("*Note: The total budget cost is always 50,000 XIRECs. Saving budget is mathematically suboptimal in this challenge format.*")

st.header("4. Breakdown of PnL")
col4, col5, col6 = st.columns(3)
col4.metric("Research PnL", f"{R_x:,.0f} XIRECs")
col5.metric("Scale PnL", f"{S_y:,.0f} XIRECs")
col6.metric("Rank Multiplier", f"{m:.3f}")
