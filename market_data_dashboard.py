from __future__ import annotations

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Install streamlit first: pip install streamlit") from exc

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_common import (
    analyse_conversion_opportunity,
    ar1_forecast,
    backtest_full_mm,
    build_product_frames,
    direction_model_summary,
    discover_local_files,
    fair_value_summary,
    find_best_exchange_cycle,
    get_limit,
    hurst_exponent,
    load_price_data,
    load_trade_data,
    mean_reversion_halflife,
    pairs_analysis,
    pacf_values,
    acf_values,
    shapiro_test_result,
    adf_test_result,
    simulate_price_paths,
    summarize_products,
    black_scholes_call,
)


st.set_page_config(page_title="IMC Market Analysis Dashboard", layout="wide")


EDGE_GRID = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10]
SKEW_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]


def _preferred_defaults(paths: list, needle: str) -> list:
    preferred = [path for path in paths if needle.lower() in str(path).lower()]
    return preferred or paths[: min(3, len(paths))]


def _series_ic(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def _product_metrics(frame: pd.DataFrame) -> dict[str, object]:
    returns = frame["mid_return"].dropna()
    fair_values = fair_value_summary(frame)
    divergence = frame["micro_price"] - frame["arith_mid"]
    next_change = frame["mid_change"].shift(-1)
    return {
        "acf1": returns.autocorr(lag=1) if not returns.empty else float("nan"),
        "hurst": hurst_exponent(frame["mid_price"]),
        "half_life": mean_reversion_halflife(frame["mid_price"]),
        "shapiro": shapiro_test_result(frame["mid_price"]),
        "adf": adf_test_result(frame["mid_price"]),
        "fair_values": fair_values,
        "obi_corr": frame["imbalance"].corr(returns.shift(-1)) if not returns.empty else float("nan"),
        "micro_ic": _series_ic(divergence, next_change),
        "ar1": ar1_forecast(frame["mid_price"]),
        "ml": direction_model_summary(frame),
    }


def _strategy_notes(product: str, metrics: dict[str, object], best_edge: float, best_skew: float) -> list[str]:
    notes: list[str] = []
    acf1 = float(metrics["acf1"])
    hurst = float(metrics["hurst"])
    half_life = float(metrics["half_life"])
    obi_corr = float(metrics["obi_corr"])
    micro_ic = float(metrics["micro_ic"])
    ml = metrics["ml"]

    if acf1 < -0.02 or hurst < 0.45:
        notes.append(f"{product} looks mean-reverting, so passive market making is the base case.")
    elif acf1 > 0.02 or hurst > 0.55:
        notes.append(f"{product} trends more than it mean-reverts, so use a faster EMA fair value and smaller passive size.")
    else:
        notes.append(f"{product} sits near random-walk behavior, so inventory control matters more than directional conviction.")

    if pd.notna(half_life):
        if half_life < 50:
            notes.append(f"Half-life is about {half_life:.1f} ticks, which supports tight quoting and quick inventory recycling.")
        elif half_life < 500:
            notes.append(f"Half-life is about {half_life:.1f} ticks, so medium-width quotes should balance fills and adverse selection.")
        else:
            notes.append(f"Half-life is long at about {half_life:.1f} ticks, so fixed fair values are risky.")

    notes.append(f"Backtest sweep likes passive edge near {best_edge} ticks and inventory skew near {best_skew:.2f}.")

    if pd.notna(obi_corr) and abs(obi_corr) > 0.05:
        notes.append(f"Order-book imbalance has predictive value ({obi_corr:+.3f}); add a 1-2 tick skew from pressure.")
    if pd.notna(micro_ic) and abs(micro_ic) > 0.05:
        notes.append(f"Microprice divergence is informative ({micro_ic:+.3f}); prefer wall-mid over arithmetic mid.")

    if pd.notna(ml["logistic_acc"]) and pd.notna(ml["baseline"]):
        edge = float(ml["logistic_acc"]) - float(ml["baseline"])
        if edge > 0.01:
            notes.append(f"Short-horizon classification has a real edge (+{edge:.3f} over baseline), which can gate aggressive trades.")

    return notes


def _price_figure(frame: pd.DataFrame, product: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["mid_price"], name="Mid", line=dict(color="#1f77b4", width=1.2)))
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["micro_price"], name="Microprice", line=dict(color="#ff7f0e", width=1.0)))
    figure.add_trace(
        go.Scattergl(
            x=frame["global_time"],
            y=frame["rolling_mean_200"],
            name="Rolling mean (200)",
            line=dict(color="#d62728", width=1.4, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=frame["global_time"],
            y=frame["ask_price_1"],
            name="Ask L1",
            line=dict(color="rgba(239,85,59,0.35)", width=0.8),
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=frame["global_time"],
            y=frame["bid_price_1"],
            name="Bid L1",
            fill="tonexty",
            line=dict(color="rgba(0,116,217,0.35)", width=0.8),
        )
    )
    figure.update_layout(
        title=f"{product} price path, spread, and fair-value overlays",
        xaxis_title="Global time",
        yaxis_title="Price",
        legend_orientation="h",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def _acf_pacf_figure(frame: pd.DataFrame, max_lag: int) -> go.Figure:
    returns = frame["mid_return"].dropna()
    acf = acf_values(returns, max_lag=max_lag)
    pacf = pacf_values(returns, max_lag=max_lag)
    lags = list(range(1, max_lag + 1))
    figure = make_subplots(rows=1, cols=2, subplot_titles=("ACF of returns", "PACF of returns"))
    figure.add_trace(go.Bar(x=lags, y=acf, marker_color=["#2ca02c" if value > 0 else "#d62728" for value in acf], name="ACF"), row=1, col=1)
    figure.add_trace(go.Bar(x=lags, y=pacf, marker_color=["#2ca02c" if value > 0 else "#d62728" for value in pacf], name="PACF"), row=1, col=2)
    figure.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return figure


def _microstructure_figure(frame: pd.DataFrame, product: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, subplot_titles=("Spread", "Order-book imbalance", "Returns", "Rolling volatility"))
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["spread"], name="Spread", line=dict(color="#7f7f7f", width=1)), row=1, col=1)
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["imbalance"], name="Imbalance", line=dict(color="#9467bd", width=1)), row=1, col=2)
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["mid_return"], name="Return", line=dict(color="#17becf", width=1)), row=2, col=1)
    figure.add_trace(go.Scattergl(x=frame["global_time"], y=frame["rolling_vol_100"], name="Rolling vol", line=dict(color="#ff9896", width=1)), row=2, col=2)
    figure.update_layout(height=560, title=f"{product} microstructure diagnostics", showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return figure


def _trade_overlay_figure(frame: pd.DataFrame, trades_df: pd.DataFrame, product: str) -> go.Figure:
    product_trades = trades_df[trades_df["symbol"] == product] if not trades_df.empty else pd.DataFrame()
    figure = go.Figure()
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["mid_price"], name="Mid", line=dict(color="#1f77b4", width=1.2)))
    if not product_trades.empty:
        figure.add_trace(
            go.Scattergl(
                x=product_trades["timestamp"],
                y=product_trades["price"],
                mode="markers",
                name="Trades",
                marker=dict(size=product_trades["quantity"].fillna(1) * 2, color="#ff7f0e", opacity=0.45),
            )
        )
    figure.update_layout(title=f"{product} market trades vs mid-price", xaxis_title="Timestamp", yaxis_title="Price", height=420, margin=dict(l=20, r=20, t=50, b=20))
    return figure


def _intraday_figure(frame: pd.DataFrame, product: str) -> go.Figure:
    avg = frame.groupby("timestamp")["intraday_demeaned"].mean()
    std = frame.groupby("timestamp")["intraday_demeaned"].std().fillna(0)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=avg.index, y=avg + std, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figure.add_trace(
        go.Scatter(
            x=avg.index,
            y=avg - std,
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.2)",
            name="Plus/minus 1 sigma",
        )
    )
    figure.add_trace(go.Scatter(x=avg.index, y=avg, name="Average demeaned path", line=dict(color="#d62728", width=1.5)))
    figure.update_layout(title=f"{product} intraday seasonality", xaxis_title="Timestamp within day", yaxis_title="Demeaned price", height=360, margin=dict(l=20, r=20, t=50, b=20))
    return figure


def _monte_carlo_figure(paths, product: str) -> go.Figure:
    p5 = np.quantile(paths, 0.05, axis=0)
    p50 = np.quantile(paths, 0.50, axis=0)
    p95 = np.quantile(paths, 0.95, axis=0)
    figure = go.Figure()
    for row in paths[: min(50, len(paths))]:
        figure.add_trace(go.Scatter(y=row, mode="lines", line=dict(color="rgba(31,119,180,0.12)", width=1), showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(y=p95, name="95th pct", line=dict(color="rgba(255,127,14,0.7)", width=1)))
    figure.add_trace(go.Scatter(y=p5, name="5th pct", fill="tonexty", line=dict(color="rgba(255,127,14,0.7)", width=1)))
    figure.add_trace(go.Scatter(y=p50, name="Median", line=dict(color="#d62728", width=2)))
    figure.update_layout(title=f"{product} Monte Carlo path envelope", xaxis_title="Forward step", yaxis_title="Simulated price", height=420, margin=dict(l=20, r=20, t=50, b=20))
    return figure


def _best_backtest(
    frame: pd.DataFrame,
    trades_df: pd.DataFrame,
    product: str,
    metrics: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, pd.DataFrame]:
    fair_values = metrics["fair_values"]
    fixed_fv = fair_values["Wall Mid"]
    use_ema = frame["mid_price"].std() > 5
    fv_method = "ema" if use_ema else "fixed"
    position_limit = get_limit(product)

    edge_rows = []
    for edge in EDGE_GRID:
        result = backtest_full_mm(
            frame,
            trades_df,
            product=product,
            fv_method=fv_method,
            fv_fixed=fixed_fv,
            passive_edge=edge,
            aggr_edge=max(1.0, edge / 2),
            position_limit=position_limit,
        )
        edge_rows.append({"edge": edge, "final_pnl": result["pnl"].iloc[-1], "max_abs_position": result["position"].abs().max()})
    edge_df = pd.DataFrame(edge_rows)
    best_edge = float(edge_df.sort_values("final_pnl", ascending=False).iloc[0]["edge"])

    skew_rows = []
    for skew in SKEW_GRID:
        result = backtest_full_mm(
            frame,
            trades_df,
            product=product,
            fv_method=fv_method,
            fv_fixed=fixed_fv,
            passive_edge=best_edge,
            aggr_edge=max(1.0, best_edge / 2),
            skew_factor=skew,
            position_limit=position_limit,
        )
        skew_rows.append({"skew": skew, "final_pnl": result["pnl"].iloc[-1], "max_abs_position": result["position"].abs().max()})
    skew_df = pd.DataFrame(skew_rows)
    best_skew = float(skew_df.sort_values("final_pnl", ascending=False).iloc[0]["skew"])

    best_run = backtest_full_mm(
        frame,
        trades_df,
        product=product,
        fv_method=fv_method,
        fv_fixed=fixed_fv,
        passive_edge=best_edge,
        aggr_edge=max(1.0, best_edge / 2),
        skew_factor=best_skew,
        position_limit=position_limit,
    )
    return edge_df, skew_df, best_edge, best_skew, best_run


def main() -> None:
    st.title("IMC Prosperity Market Analysis Dashboard")
    st.caption("Notebook-to-dashboard conversion for historical market data, signal discovery, and market-making idea generation.")

    local_price_files = discover_local_files(["**/prices_round_*.csv"])
    local_trade_files = discover_local_files(["**/trades_round_*.csv"])

    with st.sidebar:
        st.header("Data")
        selected_price_files = st.multiselect(
            "Local price files",
            local_price_files,
            default=_preferred_defaults(local_price_files, "round 1"),
            format_func=lambda path: str(path),
        )
        uploaded_price_files = st.file_uploader("Upload extra price CSVs", type=["csv"], accept_multiple_files=True)

        selected_trade_files = st.multiselect(
            "Local trade files",
            local_trade_files,
            default=_preferred_defaults(local_trade_files, "round 1"),
            format_func=lambda path: str(path),
        )
        uploaded_trade_files = st.file_uploader("Upload extra trade CSVs", type=["csv"], accept_multiple_files=True)

        max_lag = st.slider("ACF/PACF max lag", min_value=10, max_value=60, value=30, step=5)
        monte_carlo_paths = st.slider("Monte Carlo paths", min_value=100, max_value=1000, value=400, step=100)
        monte_carlo_steps = st.slider("Monte Carlo steps", min_value=100, max_value=1000, value=300, step=100)

    price_sources = list(selected_price_files) + list(uploaded_price_files or [])
    trade_sources = list(selected_trade_files) + list(uploaded_trade_files or [])

    if not price_sources:
        st.info("Choose local price files or upload price CSVs to start.")
        return

    prices_df = load_price_data(price_sources)
    trades_df = load_trade_data(trade_sources)
    product_frames = build_product_frames(prices_df)
    summary_df = summarize_products(prices_df)

    products = sorted(product_frames)
    selected_product = st.sidebar.selectbox("Product", products)
    frame = product_frames[selected_product]
    metrics = _product_metrics(frame)
    pair_pivot, pair_rows = pairs_analysis(prices_df)
    edge_df, skew_df, best_edge, best_skew, best_run = _best_backtest(frame, trades_df, selected_product, metrics)
    recommendations = _strategy_notes(selected_product, metrics, best_edge, best_skew)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", len(products))
    col2.metric("Rows", f"{len(prices_df):,}")
    col3.metric("Selected spread", f"{frame['spread'].mean():.2f}")
    col4.metric("Lag-1 ACF", f"{metrics['acf1']:.4f}")

    overview_tab, micro_tab, forecast_tab, strategy_tab, sandbox_tab = st.tabs(
        ["Overview", "Microstructure", "Forecasting", "Strategy Lab", "Idea Sandbox"]
    )

    with overview_tab:
        st.dataframe(summary_df, use_container_width=True)
        st.plotly_chart(_price_figure(frame, selected_product), use_container_width=True)
        left, right = st.columns(2)
        with left:
            dist_fig = px.histogram(frame, x="mid_price", nbins=60, title=f"{selected_product} mid-price distribution", marginal="box")
            st.plotly_chart(dist_fig, use_container_width=True)
        with right:
            spread_fig = px.histogram(frame, x="spread", nbins=50, title=f"{selected_product} spread distribution", marginal="box")
            st.plotly_chart(spread_fig, use_container_width=True)
        st.plotly_chart(_trade_overlay_figure(frame, trades_df, selected_product), use_container_width=True)

    with micro_tab:
        st.plotly_chart(_microstructure_figure(frame, selected_product), use_container_width=True)
        st.plotly_chart(_acf_pacf_figure(frame, max_lag=max_lag), use_container_width=True)
        if frame["day"].nunique() > 1:
            st.plotly_chart(_intraday_figure(frame, selected_product), use_container_width=True)

        fair_df = pd.DataFrame([metrics["fair_values"]]).T.reset_index()
        fair_df.columns = ["Method", "Value"]
        fair_df["Deviation vs Wall Mid"] = fair_df["Value"] - metrics["fair_values"]["Wall Mid"]
        st.dataframe(fair_df, use_container_width=True)

    with forecast_tab:
        shapiro = metrics["shapiro"]
        adf = metrics["adf"]
        ar1 = metrics["ar1"]
        ml = metrics["ml"]
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        stats_col1.metric("Hurst", f"{metrics['hurst']:.3f}")
        stats_col2.metric("Half-life", f"{metrics['half_life']:.1f}")
        stats_col3.metric("ADF p-value", f"{adf['p_value']:.4f}" if pd.notna(adf["p_value"]) else "n/a")
        stats_col4.metric("Shapiro p-value", f"{shapiro['p_value']:.4f}" if pd.notna(shapiro["p_value"]) else "n/a")

        ar1_actual = frame["mid_price"].tail(1000).reset_index(drop=True)
        ar1_pred = ar1["c"] + ar1["phi"] * frame["mid_price"].shift(1).tail(1000).reset_index(drop=True)
        ar1_fig = go.Figure()
        ar1_fig.add_trace(go.Scatter(y=ar1_actual, name="Actual", line=dict(color="#1f77b4", width=1.1)))
        ar1_fig.add_trace(go.Scatter(y=ar1_pred, name="AR(1)", line=dict(color="#d62728", width=1.1, dash="dash")))
        ar1_fig.update_layout(title=f"{selected_product} AR(1) forecast vs actual", height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(ar1_fig, use_container_width=True)

        ml_cols = st.columns(3)
        ml_cols[0].metric("Baseline acc", f"{ml['baseline']:.3f}" if pd.notna(ml["baseline"]) else "n/a")
        ml_cols[1].metric("Logistic acc", f"{ml['logistic_acc']:.3f}" if pd.notna(ml["logistic_acc"]) else "n/a")
        ml_cols[2].metric("MLP acc", f"{ml['mlp_acc']:.3f}" if pd.notna(ml["mlp_acc"]) else "n/a")
        if ml["coefficients"]:
            coef_df = pd.DataFrame(list(ml["coefficients"].items()), columns=["Feature", "Coefficient"])
            st.dataframe(coef_df, use_container_width=True)

        if pair_rows:
            st.dataframe(pd.DataFrame(pair_rows), use_container_width=True)
            first_pair = pair_rows[0]
            left = pair_pivot[first_pair["left"]].dropna()
            right = pair_pivot[first_pair["right"]].dropna()
            common = left.index.intersection(right.index)
            spread = left.loc[common] - first_pair["beta"] * right.loc[common]
            z = (spread - spread.rolling(200, min_periods=50).mean()) / (spread.rolling(200, min_periods=50).std() + 1e-9)
            z_fig = go.Figure()
            z_fig.add_trace(go.Scattergl(x=z.index, y=z, name="Spread z-score", line=dict(color="#9467bd", width=1.2)))
            z_fig.add_hline(y=2, line_dash="dash", line_color="#d62728")
            z_fig.add_hline(y=-2, line_dash="dash", line_color="#2ca02c")
            z_fig.update_layout(title=f"Pairs signal preview: {first_pair['pair']}", height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(z_fig, use_container_width=True)

    with strategy_tab:
        sweep_cols = st.columns(2)
        with sweep_cols[0]:
            edge_fig = px.line(edge_df, x="edge", y="final_pnl", markers=True, title=f"{selected_product} passive-edge sweep")
            edge_fig.add_vline(x=best_edge, line_dash="dash", line_color="#2ca02c")
            st.plotly_chart(edge_fig, use_container_width=True)
        with sweep_cols[1]:
            skew_fig = px.line(skew_df, x="skew", y="final_pnl", markers=True, title=f"{selected_product} skew sweep")
            skew_fig.add_vline(x=best_skew, line_dash="dash", line_color="#2ca02c")
            st.plotly_chart(skew_fig, use_container_width=True)

        best_cols = st.columns(3)
        best_cols[0].metric("Best edge", f"{best_edge}")
        best_cols[1].metric("Best skew", f"{best_skew:.2f}")
        best_cols[2].metric("Backtest PnL", f"{best_run['pnl'].iloc[-1]:.1f}")

        bt_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Backtest PnL", "Backtest inventory"))
        bt_fig.add_trace(go.Scattergl(x=best_run["global_time"], y=best_run["pnl"], name="PnL", line=dict(color="#2ca02c", width=1.5)), row=1, col=1)
        bt_fig.add_trace(go.Scattergl(x=best_run["global_time"], y=best_run["position"], name="Position", line=dict(color="#ff7f0e", width=1.2)), row=2, col=1)
        bt_fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
        st.plotly_chart(bt_fig, use_container_width=True)

        paths = simulate_price_paths(frame["mid_price"], n_paths=monte_carlo_paths, n_steps=monte_carlo_steps, method="bootstrap")
        if paths.size:
            st.plotly_chart(_monte_carlo_figure(paths, selected_product), use_container_width=True)
            final_changes = paths[:, -1] - paths[:, 0]
            risk_cols = st.columns(3)
            risk_cols[0].metric("Median path PnL", f"{pd.Series(final_changes).median():.2f}")
            risk_cols[1].metric("VaR 95%", f"{pd.Series(final_changes).quantile(0.05):.2f}")
            risk_cols[2].metric("CVaR 95%", f"{pd.Series(final_changes)[pd.Series(final_changes) <= pd.Series(final_changes).quantile(0.05)].mean():.2f}")

        st.subheader("Recommended algorithm directions")
        for note in recommendations:
            st.write(f"- {note}")

    with sandbox_tab:
        option_cols = st.columns(4)
        fv = metrics["fair_values"]["Simple Mean"]
        sigma = frame["mid_return"].dropna().std() * (10000**0.5)
        strike = option_cols[0].number_input("Strike", value=float(round(fv)))
        tenor = option_cols[1].number_input("Tenor", value=1.0, min_value=0.01, step=0.25)
        rate = option_cols[2].number_input("Risk-free rate", value=0.0, step=0.01)
        vol = option_cols[3].number_input("Volatility", value=float(max(sigma, 0.01)), min_value=0.001, step=0.01)
        option_metrics = black_scholes_call(fv, strike, tenor, rate, vol)
        st.dataframe(pd.DataFrame([option_metrics]), use_container_width=True)

        conv_cols = st.columns(7)
        conversion = analyse_conversion_opportunity(
            local_bid=conv_cols[0].number_input("Local bid", value=100.0),
            local_ask=conv_cols[1].number_input("Local ask", value=102.0),
            foreign_bid=conv_cols[2].number_input("Foreign bid", value=98.0),
            foreign_ask=conv_cols[3].number_input("Foreign ask", value=104.0),
            transport_fees=conv_cols[4].number_input("Transport", value=2.0),
            import_tariff=conv_cols[5].number_input("Import tariff", value=1.0),
            export_tariff=conv_cols[6].number_input("Export tariff", value=1.0),
        )
        st.dataframe(pd.DataFrame([conversion]), use_container_width=True)

        sample_rates = {
            ("XIRECS", "PIZZA"): 2.0,
            ("PIZZA", "WASABI"): 0.65,
            ("WASABI", "SUSHI"): 1.2,
            ("SUSHI", "XIRECS"): 0.73,
            ("XIRECS", "BAGUETTE"): 1.5,
            ("BAGUETTE", "BUTTER"): 0.52,
            ("BUTTER", "XIRECS"): 1.4,
        }
        cycle_path, cycle_profit = find_best_exchange_cycle(sample_rates, max_steps=5)
        st.write(f"Best sample exchange cycle: {' -> '.join(cycle_path)} ({cycle_profit:.3f}x)")


if __name__ == "__main__":
    main()
