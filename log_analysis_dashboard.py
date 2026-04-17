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
    backtest_full_mm,
    compute_leaderboard_metrics,
    discover_local_files,
    fair_value_summary,
    get_limit,
    load_log_bundle,
)


st.set_page_config(page_title="IMC Log Analysis Dashboard", layout="wide")

EDGE_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10]
SKEW_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
GROUP_COLORS = {"F": "#f2c94c", "M": "#f2994a", "S": "#2dd4bf", "B": "#d946ef", "I": "#facc15"}


def _preferred(paths: list, needle: str) -> list:
    preferred = [path for path in paths if needle.lower() in str(path).lower()]
    return preferred or paths[:1]


def _full_dashboard_figure(frame: pd.DataFrame, product: str, our_trades: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Price, order book, and fills",
            "Cumulative PnL",
            "Inventory",
            "Spread",
            "Imbalance",
            "Microprice vs mid",
            "Trade size timeline",
            "Fill price distribution",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["bid_price_1"], name="Bid", line=dict(color="#1f77b4", width=1)), row=1, col=1)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["ask_price_1"], name="Ask", line=dict(color="#d62728", width=1)), row=1, col=1)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["mid_price"], name="Mid", line=dict(color="#7f7f7f", width=1.2)), row=1, col=1)
    if f"{product}_fv" in frame.columns:
        figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame[f"{product}_fv"], name="FV/EMA", line=dict(color="#ff7f0e", width=1.2, dash="dash")), row=1, col=1)
    if not our_trades.empty:
        buys = our_trades[our_trades["side"] == "BUY"]
        sells = our_trades[our_trades["side"] == "SELL"]
        if not buys.empty:
            figure.add_trace(go.Scattergl(x=buys["timestamp"], y=buys["price"], mode="markers", name="Our BUY", marker=dict(color="#2ca02c", size=10, symbol="triangle-up")), row=1, col=1)
        if not sells.empty:
            figure.add_trace(go.Scattergl(x=sells["timestamp"], y=sells["price"], mode="markers", name="Our SELL", marker=dict(color="#d62728", size=10, symbol="triangle-down")), row=1, col=1)

    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["profit_and_loss"], name="PnL", line=dict(color="#2ca02c", width=1.5)), row=1, col=2)
    pos_col = f"{product}_pos"
    if pos_col in frame.columns:
        position = frame[pos_col].ffill().fillna(0)
        figure.add_trace(go.Scattergl(x=frame["timestamp"], y=position, name="Position", line=dict(color="#ff7f0e", width=1.2)), row=2, col=1)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["spread"], name="Spread", line=dict(color="#9467bd", width=1)), row=2, col=2)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["imbalance"], name="Imbalance", line=dict(color="#17becf", width=1)), row=3, col=1)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["micro_price"], name="Microprice", line=dict(color="#ff7f0e", width=1)), row=3, col=2)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["mid_price"], name="Mid ref", line=dict(color="#7f7f7f", width=1, dash="dot")), row=3, col=2)
    if not our_trades.empty:
        figure.add_trace(go.Bar(x=our_trades["timestamp"], y=our_trades["quantity"], name="Trade qty", marker_color=["#2ca02c" if side == "BUY" else "#d62728" for side in our_trades["side"]]), row=4, col=1)
        figure.add_trace(go.Histogram(x=our_trades["price"], name="Fill prices", marker_color="#1f77b4", opacity=0.75), row=4, col=2)

    figure.update_layout(height=1100, title=f"{product} execution dashboard", legend_orientation="h", margin=dict(l=20, r=20, t=70, b=20))
    return figure


def _price_impact(frame: pd.DataFrame, trades: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    indexed = frame.set_index("timestamp")
    timeline = indexed.index.tolist()
    buy_paths = []
    sell_paths = []

    for trade in trades.itertuples(index=False):
        if trade.timestamp not in indexed.index:
            continue
        current_index = timeline.index(trade.timestamp)
        fill_mid = float(indexed.loc[trade.timestamp, "mid_price"])
        path = []
        for step in range(lookahead):
            if current_index + step >= len(timeline):
                path.append(np.nan)
            else:
                future_mid = float(indexed.loc[timeline[current_index + step], "mid_price"])
                path.append(future_mid - fill_mid)
        if trade.side == "BUY":
            buy_paths.append(path)
        else:
            sell_paths.append(path)

    rows = []
    for step in range(lookahead):
        rows.append(
            {
                "step": step,
                "buy_impact": np.nanmean([path[step] for path in buy_paths]) if buy_paths else np.nan,
                "sell_impact": np.nanmean([path[step] for path in sell_paths]) if sell_paths else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fill_summary(frame: pd.DataFrame, product: str, our_trades: pd.DataFrame) -> dict[str, float]:
    summary = {
        "fills": float(len(our_trades)),
        "fill_rate": float(len(our_trades) / max(len(frame), 1)),
        "avg_fill": float(our_trades["quantity"].mean()) if not our_trades.empty else 0.0,
        "avg_buy": np.nan,
        "avg_sell": np.nan,
        "round_trip_pnl": np.nan,
    }
    if our_trades.empty:
        return summary
    buys = our_trades[our_trades["side"] == "BUY"]
    sells = our_trades[our_trades["side"] == "SELL"]
    if not buys.empty:
        summary["avg_buy"] = float((buys["price"] * buys["quantity"]).sum() / buys["quantity"].sum())
    if not sells.empty:
        summary["avg_sell"] = float((sells["price"] * sells["quantity"]).sum() / sells["quantity"].sum())
    if not buys.empty and not sells.empty:
        matched_volume = min(buys["quantity"].sum(), sells["quantity"].sum())
        summary["round_trip_pnl"] = float((summary["avg_sell"] - summary["avg_buy"]) * matched_volume)
    return summary


def _quote_stats(frame: pd.DataFrame, product: str) -> dict[str, float]:
    bid_col = f"{product}_bid"
    ask_col = f"{product}_ask"
    if bid_col not in frame.columns or ask_col not in frame.columns:
        return {}
    visible = frame.dropna(subset=[bid_col, ask_col])
    if visible.empty:
        return {}
    inside = (visible[bid_col] > visible["bid_price_1"]) | (visible[ask_col] < visible["ask_price_1"])
    fv_col = f"{product}_fv"
    bias = (visible[fv_col] - visible["mid_price"]).mean() if fv_col in visible.columns else np.nan
    return {
        "inside_share": float(inside.mean()),
        "our_spread": float((visible[ask_col] - visible[bid_col]).mean()),
        "bot_spread": float(visible["spread"].mean()),
        "fv_bias": float(bias) if pd.notna(bias) else np.nan,
    }


def _recommendations(product: str, fill_summary: dict[str, float], quote_stats: dict[str, float], impact_df: pd.DataFrame, leaderboard: dict[str, object]) -> list[str]:
    notes: list[str] = []
    if fill_summary["fill_rate"] < 0.05:
        notes.append(f"{product}: fill rate is very low, so quotes are probably too far from the bot's price levels.")
    if pd.notna(fill_summary["round_trip_pnl"]) and fill_summary["round_trip_pnl"] < 0:
        notes.append(f"{product}: round-trip PnL is negative, so widen the passive edge before chasing more fills.")
    if quote_stats and quote_stats["inside_share"] < 0.1:
        notes.append(f"{product}: quotes rarely sit inside the bot spread, so the strategy is not earning queue priority.")
    if quote_stats and pd.notna(quote_stats["fv_bias"]) and abs(quote_stats["fv_bias"]) > 2:
        notes.append(f"{product}: fair value bias is materially off-market ({quote_stats['fv_bias']:+.2f}); speed up or stabilize the FV model.")
    if not impact_df.empty and pd.notna(impact_df["buy_impact"].iloc[-1]) and impact_df["buy_impact"].iloc[-1] < -0.5:
        notes.append(f"{product}: buy fills are followed by price drops, which is a classic adverse-selection signal.")
    if leaderboard and leaderboard.get("recovery_ratio", 0) < 5:
        notes.append("Portfolio-level recovery ratio is weak, so add volatility gating or drawdown circuit breakers.")
    return notes


def _snapshot_tables(bundle, product: str, timestamp: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = bundle.product_frames[product]
    orderbook = frame[frame["timestamp"] <= timestamp].tail(1).copy()
    if orderbook.empty:
        orderbook = frame.head(1).copy()

    orderbook_table = pd.DataFrame(
        [
            {"side": "Ask L3", "price": orderbook.iloc[0].get("ask_price_3"), "volume": orderbook.iloc[0].get("ask_volume_3")},
            {"side": "Ask L2", "price": orderbook.iloc[0].get("ask_price_2"), "volume": orderbook.iloc[0].get("ask_volume_2")},
            {"side": "Ask L1", "price": orderbook.iloc[0].get("ask_price_1"), "volume": orderbook.iloc[0].get("ask_volume_1")},
            {"side": "Bid L1", "price": orderbook.iloc[0].get("bid_price_1"), "volume": orderbook.iloc[0].get("bid_volume_1")},
            {"side": "Bid L2", "price": orderbook.iloc[0].get("bid_price_2"), "volume": orderbook.iloc[0].get("bid_volume_2")},
            {"side": "Bid L3", "price": orderbook.iloc[0].get("bid_price_3"), "volume": orderbook.iloc[0].get("bid_volume_3")},
        ]
    )

    our_trades = bundle.our_trades[bundle.our_trades["timestamp"] == timestamp] if not bundle.our_trades.empty else pd.DataFrame()
    market_trades = bundle.trades[(bundle.trades["timestamp"] == timestamp) & (~bundle.trades["is_our_trade"])] if not bundle.trades.empty else pd.DataFrame()
    return orderbook_table, our_trades, market_trades


def _friendly_label(column: str, product: str) -> str:
    return column.removeprefix(f"{product}_").replace("_", " ").title()


def _is_price_like(column: str) -> bool:
    return any(token in column.lower() for token in ("price", "mid", "fv", "ema", "bid", "ask", "wall"))


def _sample_rows(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, max_points, dtype=int)
    return frame.iloc[np.unique(indices)].copy()


def _lookup_row(frame: pd.DataFrame, timestamp: int) -> pd.Series:
    exact = frame[frame["timestamp"] == timestamp]
    if not exact.empty:
        return exact.iloc[-1]
    return frame.iloc[int((frame["timestamp"] - timestamp).abs().idxmin())]


def _build_position_series(frame: pd.DataFrame, product: str, our_trades: pd.DataFrame) -> pd.Series:
    parsed_col = f"{product}_pos"
    if parsed_col in frame.columns and frame[parsed_col].notna().any():
        return frame[parsed_col].ffill().fillna(0)
    if our_trades.empty:
        return pd.Series(0.0, index=frame.index)
    signed = np.where(our_trades["side"] == "BUY", our_trades["quantity"], -our_trades["quantity"])
    cumulative = our_trades.assign(signed_qty=signed).groupby("timestamp")["signed_qty"].sum().cumsum()
    return frame["timestamp"].map(cumulative).ffill().fillna(0)


def _prepare_trade_view(frame: pd.DataFrame, trades: pd.DataFrame, product: str, lookahead: int = 10) -> pd.DataFrame:
    product_trades = trades[trades["symbol"] == product].copy() if not trades.empty else pd.DataFrame()
    if product_trades.empty:
        return product_trades

    market_view = frame[["timestamp", "mid_price", "bid_price_1", "ask_price_1", "spread"]].copy()
    market_view["future_mid"] = market_view["mid_price"].shift(-lookahead)
    product_trades = product_trades.merge(market_view, on="timestamp", how="left")
    price = pd.to_numeric(product_trades["price"], errors="coerce")
    bid_touch = pd.to_numeric(product_trades["bid_price_1"], errors="coerce")
    ask_touch = pd.to_numeric(product_trades["ask_price_1"], errors="coerce")
    spread = pd.to_numeric(product_trades["spread"], errors="coerce").fillna(0)
    future_move = pd.to_numeric(product_trades["future_mid"], errors="coerce") - pd.to_numeric(product_trades["mid_price"], errors="coerce")

    product_trades["aggressor_side"] = np.select(
        [price >= ask_touch - 1e-9, price <= bid_touch + 1e-9],
        ["BUY", "SELL"],
        default="UNKNOWN",
    )
    product_trades["style"] = np.where(
        product_trades["is_our_trade"],
        "own",
        np.where(product_trades["aggressor_side"].isin(["BUY", "SELL"]), "taker", "maker"),
    )

    taker_quantities = product_trades.loc[product_trades["style"] == "taker", "quantity"].dropna()
    big_threshold = float(taker_quantities.quantile(0.75)) if not taker_quantities.empty else float(product_trades["quantity"].max())
    informed = (
        ((product_trades["aggressor_side"] == "BUY") & (future_move > np.where(spread > 0, spread / 4, 0.5)))
        | ((product_trades["aggressor_side"] == "SELL") & (future_move < -np.where(spread > 0, spread / 4, 0.5)))
    )
    product_trades["group"] = np.select(
        [product_trades["is_our_trade"], informed.fillna(False), product_trades["style"] == "maker", product_trades["quantity"] >= big_threshold],
        ["F", "I", "M", "B"],
        default="S",
    )

    buyer = product_trades["buyer"].fillna("").astype(str).str.strip()
    seller = product_trades["seller"].fillna("").astype(str).str.strip()
    product_trades["trader_key"] = np.select(
        [product_trades["is_our_trade"] & buyer.ne(""), product_trades["is_our_trade"] & seller.ne(""), buyer.ne("") & seller.ne(""), buyer.ne(""), seller.ne("")],
        ["cp:" + buyer, "cp:" + seller, buyer + "->" + seller, buyer, seller],
        default=np.where(product_trades["style"] == "taker", "market_taker", "market_maker"),
    )
    product_trades["future_move"] = future_move

    trader_ids = []
    for group, group_frame in product_trades.groupby("group", sort=False):
        key_map = {key: f"{group}{idx}" for idx, key in enumerate(sorted(group_frame["trader_key"].astype(str).unique()), start=1)}
        trader_ids.extend(group_frame["trader_key"].map(key_map))
    product_trades["trader_id"] = trader_ids
    product_trades["hover_text"] = (
        "TS " + product_trades["timestamp"].astype(str)
        + "<br>Price " + product_trades["price"].astype(str)
        + "<br>Qty " + product_trades["quantity"].astype(str)
        + "<br>Buyer " + buyer.replace("", "-")
        + "<br>Seller " + seller.replace("", "-")
        + "<br>Group " + product_trades["group"] + " / " + product_trades["trader_id"]
    )
    return product_trades.sort_values(["timestamp", "price"]).reset_index(drop=True)


def _filter_trade_view(trades: pd.DataFrame, groups: list[str], trader_ids: list[str], qty_range: tuple[int, int]) -> pd.DataFrame:
    if trades.empty:
        return trades
    filtered = trades[trades["group"].isin(groups)].copy()
    filtered = filtered[filtered["quantity"].between(qty_range[0], qty_range[1])]
    if trader_ids:
        filtered = filtered[filtered["trader_id"].isin(trader_ids)]
    return filtered


def _normalize(frame: pd.DataFrame, column: str, normalize_by: str | None) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    if normalize_by and normalize_by in frame.columns and _is_price_like(column):
        return series - pd.to_numeric(frame[normalize_by], errors="coerce")
    return series


def _build_order_book_figure(frame: pd.DataFrame, trades: pd.DataFrame, product: str, selected_ts: int, levels: list[int], overlays: list[str], normalize_by: str | None) -> go.Figure:
    figure = go.Figure()
    level_opacity = {1: 0.95, 2: 0.65, 3: 0.42}
    for level in levels:
        for side, color in [("bid", "rgba(38, 99, 235, 1.0)"), ("ask", "rgba(220, 38, 38, 1.0)")]:
            price_col = f"{side}_price_{level}"
            volume_col = f"{side}_volume_{level}"
            if price_col not in frame.columns or frame[price_col].notna().sum() == 0:
                continue
            figure.add_trace(
                go.Scattergl(
                    x=frame["timestamp"],
                    y=_normalize(frame, price_col, normalize_by),
                    mode="markers",
                    name=f"{side.title()} L{level}",
                    marker=dict(color=color, size=np.clip(pd.to_numeric(frame.get(volume_col, 1), errors="coerce").fillna(1) * 1.5, 4, 18), opacity=level_opacity.get(level, 0.35), symbol="square"),
                    customdata=pd.to_numeric(frame.get(volume_col, 0), errors="coerce").fillna(0),
                    hovertemplate=f"TS %{{x}}<br>{side.title()} L{level} %{{y:.2f}}<br>Size %{{customdata:.0f}}<extra></extra>",
                )
            )
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=_normalize(frame, "mid_price", normalize_by), mode="lines", name="Mid", line=dict(color="#6b7280", width=1.4)))
    for overlay in overlays:
        if overlay in frame.columns:
            figure.add_trace(go.Scattergl(x=frame["timestamp"], y=_normalize(frame, overlay, normalize_by), mode="lines", name=_friendly_label(overlay, product), line=dict(width=2, dash="dash")))
    symbol_map = {"F": "x", "M": "square", "S": "triangle-up", "B": "triangle-down", "I": "diamond"}
    reference = frame.set_index("timestamp")[normalize_by] if normalize_by and normalize_by in frame.columns else pd.Series(dtype=float)
    for group in ["F", "M", "S", "B", "I"]:
        group_trades = trades[trades["group"] == group]
        if group_trades.empty:
            continue
        y_values = pd.to_numeric(group_trades["price"], errors="coerce")
        if not reference.empty:
            y_values = y_values - group_trades["timestamp"].map(reference).fillna(0)
        figure.add_trace(
            go.Scattergl(
                x=group_trades["timestamp"],
                y=y_values,
                mode="markers",
                name=f"{group} trades",
                marker=dict(color=GROUP_COLORS[group], size=np.clip(group_trades["quantity"].fillna(1) * 1.8, 8, 20), opacity=0.9 if group == "F" else 0.7, symbol=symbol_map[group], line=dict(width=1, color="#111827")),
                text=group_trades["hover_text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figure.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
    figure.update_layout(height=560, margin=dict(l=20, r=20, t=55, b=20), legend_orientation="h", title=f"{product} order book and trades", xaxis_title="Timestamp", yaxis_title="Relative price" if normalize_by else "Price")
    return figure


def _build_pnl_figure(frame: pd.DataFrame, leaderboard: dict[str, object], product: str, selected_ts: int) -> go.Figure:
    figure = go.Figure()
    pnl_df = leaderboard.get("pnl_df", pd.DataFrame())
    if not pnl_df.empty:
        figure.add_trace(go.Scattergl(x=pnl_df["timestamp"], y=pnl_df["total"], name="Total", line=dict(color="#111827", width=2)))
        if product in pnl_df.columns:
            figure.add_trace(go.Scattergl(x=pnl_df["timestamp"], y=pnl_df[product], name=product, line=dict(color="#ef4444", width=1.6)))
    else:
        figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["profit_and_loss"], name=product, line=dict(color="#ef4444", width=1.6)))
    figure.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
    figure.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), title="PnL performance", xaxis_title="Timestamp", yaxis_title="PnL")
    return figure


def _build_position_figure(frame: pd.DataFrame, position: pd.Series, product: str, selected_ts: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=position, name=product, line=dict(color="#4f46e5", width=2, shape="hv")))
    figure.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
    figure.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), title=f"Position: {product}", xaxis_title="Timestamp", yaxis_title="Position")
    return figure


def _build_depth_figure(frame: pd.DataFrame, selected_ts: int) -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Spread", "Liquidity"))
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["spread"], name="Spread", line=dict(color="#8b5cf6", width=1.5)), row=1, col=1)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["bid_depth"], name="Bid depth", line=dict(color="#10b981", width=1.4)), row=1, col=2)
    figure.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["ask_depth"], name="Ask depth", line=dict(color="#ef4444", width=1.4)), row=1, col=2)
    figure.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
    figure.update_layout(height=330, margin=dict(l=20, r=20, t=45, b=20), legend_orientation="h")
    return figure


def _build_indicator_figure(frame: pd.DataFrame, product: str, selected_ts: int, indicators: list[str], normalize_by: str | None) -> go.Figure:
    figure = go.Figure()
    for column in indicators:
        if column in frame.columns:
            figure.add_trace(go.Scattergl(x=frame["timestamp"], y=_normalize(frame, column, normalize_by if _is_price_like(column) else None), name=_friendly_label(column, product), line=dict(width=1.8)))
    figure.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
    figure.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20), title="Indicator overlays", xaxis_title="Timestamp")
    return figure


def _build_snapshot_depth_figure(snapshot: pd.Series) -> go.Figure:
    rows = []
    for level in [3, 2, 1]:
        price = snapshot.get(f"bid_price_{level}")
        volume = snapshot.get(f"bid_volume_{level}")
        if pd.notna(price) and pd.notna(volume):
            rows.append((f"{price:.1f}", -float(volume)))
    for level in [1, 2, 3]:
        price = snapshot.get(f"ask_price_{level}")
        volume = snapshot.get(f"ask_volume_{level}")
        if pd.notna(price) and pd.notna(volume):
            rows.append((f"{price:.1f}", float(volume)))
    labels = [label for label, _ in rows]
    values = [value for _, value in rows]
    colors = ["#10b981" if value < 0 else "#ef4444" for value in values]
    figure = go.Figure([go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[abs(value) for value in values], textposition="auto")])
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), title="Order book snapshot", xaxis_title="Bid size < 0 | Ask size > 0", yaxis_title="Price")
    return figure


def _metric_value_at_timestamp(leaderboard: dict[str, object], product: str, timestamp: int) -> tuple[float, float]:
    pnl_df = leaderboard.get("pnl_df", pd.DataFrame())
    if pnl_df.empty:
        return np.nan, np.nan
    row = pnl_df.iloc[int((pnl_df["timestamp"] - timestamp).abs().idxmin())]
    return float(row.get("total", np.nan)), float(row.get(product, np.nan))


def main() -> None:
    st.title("IMC Prosperity Log Analysis Dashboard")
    st.caption("Microstructure-first execution review, synced timestamp inspection, log overlays, and tuning sweeps.")

    local_json_files = discover_local_files(["**/*.json"])
    local_log_files = discover_local_files(["**/*.log"])
    default_log = _preferred(local_log_files, "v3")

    with st.sidebar:
        st.header("Bundle")
        selected_json = st.selectbox("Local JSON result", [None] + local_json_files, index=1 if local_json_files else 0, format_func=lambda value: "None" if value is None else str(value))
        selected_log = st.selectbox("Local LOG payload", [None] + local_log_files, index=(local_log_files.index(default_log[0]) + 1) if default_log and default_log[0] in local_log_files else (1 if local_log_files else 0), format_func=lambda value: "None" if value is None else str(value))
        uploaded_json = st.file_uploader("Upload JSON result", type=["json"])
        uploaded_log = st.file_uploader("Upload LOG payload", type=["log", "json"])
        lookahead = st.slider("Post-fill lookahead", min_value=5, max_value=50, value=20, step=5)

    json_source = uploaded_json or selected_json
    log_source = uploaded_log or selected_log
    if json_source is None and log_source is None:
        st.info("Choose a local result file or upload a JSON/LOG pair to begin.")
        return

    bundle = load_log_bundle(json_source=json_source, log_source=log_source)
    if bundle.market.empty:
        st.error("No activities log was found in the supplied files.")
        return

    leaderboard = compute_leaderboard_metrics(bundle.market, bundle.product_frames, bundle.our_trades)
    selected_product = st.sidebar.selectbox("Product", bundle.products)
    frame = bundle.product_frames[selected_product].copy()
    product_trades = bundle.our_trades[bundle.our_trades["symbol"] == selected_product].copy() if not bundle.our_trades.empty else pd.DataFrame()
    position = _build_position_series(frame, selected_product, product_trades)
    trade_view = _prepare_trade_view(frame, bundle.trades, selected_product, lookahead=max(3, lookahead // 2))

    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col]) and col not in {"day", "global_time"}]
    price_overlay_options = [col for col in numeric_cols if _is_price_like(col) and col not in {"timestamp", "bid_price_1", "bid_price_2", "bid_price_3", "ask_price_1", "ask_price_2", "ask_price_3"}]
    indicator_options = [col for col in numeric_cols if col != "timestamp"]
    default_overlays = [col for col in [f"{selected_product}_fv", f"{selected_product}_ema", "wall_mid", "micro_price"] if col in price_overlay_options][:2]

    with st.sidebar:
        st.header("Visualization")
        normalize_choices = [None] + price_overlay_options
        normalize_by = st.selectbox("Normalize prices by", normalize_choices, format_func=lambda value: "None" if value is None else _friendly_label(value, selected_product))
        overlays = st.multiselect("Price overlays", price_overlay_options, default=default_overlays, format_func=lambda value: _friendly_label(value, selected_product))
        indicators = st.multiselect("Indicator lens", indicator_options, default=[col for col in [f"{selected_product}_fv", f"{selected_product}_ema", "imbalance", "spread"] if col in indicator_options][:3], format_func=lambda value: _friendly_label(value, selected_product))
        show_levels = st.multiselect("Order book levels", [1, 2, 3], default=[1, 2])
        render_mode = st.radio("Render mode", ["Sampled", "Full"], horizontal=True)
        max_points = st.slider("Visible ticks", min_value=300, max_value=3000, value=900, step=100)
        max_trade_markers = st.slider("Trade markers cap", min_value=100, max_value=2000, value=600, step=100)

    qty_max = int(trade_view["quantity"].max()) if not trade_view.empty else 1
    groups = [group for group in ["F", "M", "S", "B", "I"] if trade_view.empty or group in trade_view["group"].unique()]
    traders = sorted(trade_view["trader_id"].unique().tolist()) if not trade_view.empty else []
    with st.sidebar:
        st.header("Trade filters")
        enabled_groups = st.multiselect("Groups", groups, default=groups)
        trader_filter = st.multiselect("Specific traders", traders, default=traders[: min(8, len(traders))] if len(traders) <= 8 else [])
        qty_range = st.slider("Quantity range", min_value=0, max_value=max(qty_max, 1), value=(0, max(qty_max, 1)))

    timestamps = frame["timestamp"].astype(int).tolist()
    ts_key = f"log_dashboard_ts::{selected_product}"
    if ts_key not in st.session_state:
        st.session_state[ts_key] = max(0, len(timestamps) // 2)
    nav_step = st.radio("Playback step", [1, 2, 5, 10, 20], horizontal=True, index=0)
    nav_cols = st.columns([0.8, 0.8, 1.2, 5.0, 1.7, 1.0])
    if nav_cols[0].button("Prev", use_container_width=True):
        st.session_state[ts_key] = max(0, st.session_state[ts_key] - nav_step)
    if nav_cols[1].button("Next", use_container_width=True):
        st.session_state[ts_key] = min(len(timestamps) - 1, st.session_state[ts_key] + nav_step)
    st.session_state[ts_key] = nav_cols[3].slider("Tick", min_value=0, max_value=max(len(timestamps) - 1, 0), value=st.session_state[ts_key], label_visibility="collapsed")
    go_to_ts = nav_cols[4].number_input("Go to TS", min_value=int(timestamps[0]), max_value=int(timestamps[-1]), value=int(timestamps[st.session_state[ts_key]]), step=100, label_visibility="collapsed")
    if nav_cols[5].button("Jump", use_container_width=True):
        st.session_state[ts_key] = int((pd.Series(timestamps) - int(go_to_ts)).abs().idxmin())

    selected_idx = st.session_state[ts_key]
    selected_ts = int(timestamps[selected_idx])
    row = _lookup_row(frame, selected_ts)
    total_pnl, product_pnl = _metric_value_at_timestamp(leaderboard, selected_product, selected_ts)
    current_position = float(position.iloc[int((frame["timestamp"] - selected_ts).abs().idxmin())]) if not position.empty else 0.0

    filtered_trades = _filter_trade_view(trade_view, enabled_groups or groups, trader_filter, qty_range)
    if len(filtered_trades) > max_trade_markers:
        filtered_trades = _sample_rows(filtered_trades, max_trade_markers)
    frame_view = _sample_rows(frame, max_points) if render_mode == "Sampled" else frame.copy()

    metrics = st.columns(5)
    metrics[0].metric("Total PnL", f"{total_pnl:.2f}" if pd.notna(total_pnl) else "n/a")
    metrics[1].metric("Max drawdown", f"{leaderboard.get('max_drawdown', float('nan')):.2f}")
    metrics[2].metric(f"{selected_product} PnL", f"{product_pnl:.2f}" if pd.notna(product_pnl) else "n/a")
    metrics[3].metric("Position", f"{current_position:+.0f}")
    metrics[4].metric("Microprice", f"{row.get('micro_price', np.nan):.2f}" if pd.notna(row.get("micro_price", np.nan)) else "n/a")
    st.caption(f"Tick {selected_idx + 1} / {len(timestamps)} | TS {selected_ts} | Day {int(row.get('day', 0))} | {100 * selected_idx / max(len(timestamps) - 1, 1):.1f}% through replay | {len(frame_view):,} / {len(frame):,} ticks plotted | {len(filtered_trades):,} trade markers")

    market_tab, snapshot_tab, execution_tab, logs_tab, sim_tab = st.tabs(["Market View", "Snapshot", "Execution", "Logs & Signals", "Simulation"])

    with market_tab:
        st.plotly_chart(_build_order_book_figure(frame_view, filtered_trades, selected_product, selected_ts, show_levels or [1], overlays, normalize_by), use_container_width=True)
        lower_cols = st.columns(2)
        lower_cols[0].plotly_chart(_build_pnl_figure(frame, leaderboard, selected_product, selected_ts), use_container_width=True)
        lower_cols[1].plotly_chart(_build_position_figure(frame, position, selected_product, selected_ts), use_container_width=True)
        st.plotly_chart(_build_depth_figure(frame_view, selected_ts), use_container_width=True)

    with snapshot_tab:
        snapshot_table, own_at_ts, market_at_ts = _snapshot_tables(bundle, selected_product, selected_ts)
        snap_cols = st.columns([1.2, 1.8])
        snap_cols[0].dataframe(snapshot_table.dropna(how="all"), use_container_width=True)
        snap_cols[1].plotly_chart(_build_snapshot_depth_figure(row), use_container_width=True)
        summary_cols = st.columns(4)
        summary_cols[0].metric("Spread", f"{row.get('spread', np.nan):.2f}" if pd.notna(row.get("spread", np.nan)) else "n/a")
        summary_cols[1].metric("Wall mid", f"{row.get('wall_mid', np.nan):.2f}" if pd.notna(row.get("wall_mid", np.nan)) else "n/a")
        summary_cols[2].metric("Pressure", f"{100 * row.get('imbalance', np.nan):+.1f}%" if pd.notna(row.get("imbalance", np.nan)) else "n/a")
        summary_cols[3].metric("Product PnL", f"{row.get('profit_and_loss', np.nan):.2f}" if pd.notna(row.get("profit_and_loss", np.nan)) else "n/a")
        show_fills_through = st.checkbox(f"Show all own fills through {selected_ts}", value=False)
        own_view = product_trades[product_trades["timestamp"] <= selected_ts] if show_fills_through else own_at_ts
        detail_cols = st.columns(3)
        detail_cols[0].dataframe(own_view if not own_view.empty else pd.DataFrame({"info": ["No own fills at this view"]}), use_container_width=True)
        detail_cols[1].dataframe(market_at_ts if not market_at_ts.empty else pd.DataFrame({"info": ["No market trades at this timestamp"]}), use_container_width=True)
        product_fields = ["timestamp"] + [col for col in frame.columns if col.startswith(f"{selected_product}_") and not col.endswith("_log_line")]
        detail_cols[2].dataframe(pd.DataFrame([row[product_fields]]).T.rename(columns={0: "value"}), use_container_width=True)

    with execution_tab:
        fill_summary = _fill_summary(frame, selected_product, product_trades)
        quote_stats = _quote_stats(frame, selected_product)
        impact_df = _price_impact(frame, product_trades, lookahead=lookahead)
        recs = _recommendations(selected_product, fill_summary, quote_stats, impact_df, leaderboard)
        fill_cols = st.columns(4)
        fill_cols[0].metric("Fill rate", f"{100 * fill_summary['fill_rate']:.1f}%")
        fill_cols[1].metric("Avg buy", f"{fill_summary['avg_buy']:.2f}" if pd.notna(fill_summary["avg_buy"]) else "n/a")
        fill_cols[2].metric("Avg sell", f"{fill_summary['avg_sell']:.2f}" if pd.notna(fill_summary["avg_sell"]) else "n/a")
        fill_cols[3].metric("Round-trip est.", f"{fill_summary['round_trip_pnl']:.2f}" if pd.notna(fill_summary["round_trip_pnl"]) else "n/a")

        exec_cols = st.columns(2)
        trade_fig = go.Figure()
        other_trades = filtered_trades[~filtered_trades["is_our_trade"]]
        if not other_trades.empty:
            trade_fig.add_trace(go.Scattergl(x=other_trades["timestamp"], y=other_trades["price"], mode="markers", name="Market trades", marker=dict(size=np.clip(other_trades["quantity"].fillna(1) * 1.6, 6, 18), opacity=0.35, color="#2563eb"), text=other_trades["hover_text"], hovertemplate="%{text}<extra></extra>"))
        if not product_trades.empty:
            trade_fig.add_trace(go.Scattergl(x=product_trades["timestamp"], y=product_trades["price"], mode="markers", name="Our fills", marker=dict(size=np.clip(product_trades["quantity"].fillna(1) * 2, 8, 20), opacity=0.75, color="#f59e0b", symbol="x")))
        trade_fig.add_trace(go.Scattergl(x=frame["timestamp"], y=frame["mid_price"], name="Mid", line=dict(color="#6b7280", width=1.2)))
        trade_fig.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
        trade_fig.update_layout(height=380, margin=dict(l=20, r=20, t=45, b=20), title="Trades vs mid")
        exec_cols[0].plotly_chart(trade_fig, use_container_width=True)

        inventory_fig = make_subplots(rows=1, cols=2, subplot_titles=("Inventory path", "Inventory vs PnL"))
        inventory_fig.add_trace(go.Scattergl(x=frame["timestamp"], y=position, name="Position", line=dict(color="#4f46e5", width=1.8)), row=1, col=1)
        inventory_fig.add_trace(go.Scattergl(x=position, y=frame["profit_and_loss"], mode="markers", name="PnL scatter", marker=dict(size=5, color=frame["timestamp"], colorscale="Plasma", opacity=0.55)), row=1, col=2)
        inventory_fig.update_layout(height=380, margin=dict(l=20, r=20, t=45, b=20), showlegend=False)
        exec_cols[1].plotly_chart(inventory_fig, use_container_width=True)

        if not impact_df.empty:
            impact_fig = go.Figure()
            if impact_df["buy_impact"].notna().any():
                impact_fig.add_trace(go.Scatter(x=impact_df["step"], y=impact_df["buy_impact"], name="After BUY", line=dict(color="#16a34a", width=2)))
            if impact_df["sell_impact"].notna().any():
                impact_fig.add_trace(go.Scatter(x=impact_df["step"], y=impact_df["sell_impact"], name="After SELL", line=dict(color="#dc2626", width=2)))
            impact_fig.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20), title="Post-fill impact", xaxis_title="Ticks after fill", yaxis_title="Mid change")
            st.plotly_chart(impact_fig, use_container_width=True)

        quote_cols = st.columns(4)
        quote_cols[0].metric("Inside share", f"{100 * quote_stats.get('inside_share', float('nan')):.1f}%" if quote_stats else "n/a")
        quote_cols[1].metric("Our spread", f"{quote_stats.get('our_spread', float('nan')):.2f}" if quote_stats else "n/a")
        quote_cols[2].metric("Bot spread", f"{quote_stats.get('bot_spread', float('nan')):.2f}" if quote_stats else "n/a")
        quote_cols[3].metric("FV bias", f"{quote_stats.get('fv_bias', float('nan')):+.2f}" if quote_stats and pd.notna(quote_stats.get("fv_bias")) else "n/a")
        st.subheader("Improvement roadmap")
        if recs:
            for note in recs:
                st.write(f"- {note}")
        else:
            st.write("- Metrics look healthy. The next step is robustness testing across multiple submissions and days.")

    with logs_tab:
        st.plotly_chart(_build_indicator_figure(frame_view, selected_product, selected_ts, indicators, normalize_by), use_container_width=True)
        log_cols = st.columns([1.2, 1.8])
        if not bundle.parsed_logs.empty:
            parsed_row = bundle.parsed_logs.iloc[int((bundle.parsed_logs["timestamp"] - selected_ts).abs().idxmin())]
            parsed_table = pd.DataFrame(parsed_row.dropna()).reset_index()
            parsed_table.columns = ["field", "value"]
            log_cols[0].dataframe(parsed_table, use_container_width=True)
        else:
            log_cols[0].info("No parsed lambda logs were available in this submission.")
        raw_tabs = log_cols[1].tabs(["Algorithm logs", "Sandbox logs", "Trader data"])
        if not bundle.logs.empty:
            raw_log = bundle.logs.iloc[int((bundle.logs["timestamp"] - selected_ts).abs().idxmin())]
            raw_tabs[0].text_area("Lambda log", value=str(raw_log.get("lambdaLog", "")), height=240)
            raw_tabs[1].text_area("Sandbox log", value=str(raw_log.get("sandboxLog", "")), height=240)
        else:
            raw_tabs[0].info("No lambda logs stored in this payload.")
            raw_tabs[1].info("No sandbox logs stored in this payload.")
        raw_tabs[2].dataframe(trade_view[trade_view["timestamp"] == selected_ts][["timestamp", "price", "quantity", "group", "trader_id", "buyer", "seller", "future_move"]] if not trade_view.empty and (trade_view["timestamp"] == selected_ts).any() else pd.DataFrame({"info": ["No trader prints at this timestamp"]}), use_container_width=True)

    with sim_tab:
        fair_values = fair_value_summary(frame)
        use_ema = frame["mid_price"].std() > 5
        edge_rows = []
        for edge in EDGE_GRID:
            run = backtest_full_mm(frame, bundle.trades, product=selected_product, fv_method="ema" if use_ema else "fixed", fv_fixed=fair_values["Wall Mid"], passive_edge=edge, aggr_edge=max(1, edge / 2), position_limit=get_limit(selected_product))
            edge_rows.append({"edge": edge, "final_pnl": run["pnl"].iloc[-1] if not run.empty else np.nan})
        edge_df = pd.DataFrame(edge_rows)
        best_edge = float(edge_df.sort_values("final_pnl", ascending=False).iloc[0]["edge"]) if not edge_df.empty else 4.0
        skew_rows = []
        for skew in SKEW_GRID:
            run = backtest_full_mm(frame, bundle.trades, product=selected_product, fv_method="ema" if use_ema else "fixed", fv_fixed=fair_values["Wall Mid"], passive_edge=best_edge, aggr_edge=max(1, best_edge / 2), skew_factor=skew, position_limit=get_limit(selected_product))
            skew_rows.append({"skew": skew, "final_pnl": run["pnl"].iloc[-1] if not run.empty else np.nan})
        skew_df = pd.DataFrame(skew_rows)
        sim_cols = st.columns(2)
        sim_cols[0].plotly_chart(px.line(edge_df, x="edge", y="final_pnl", markers=True, title=f"{selected_product} edge sweep"), use_container_width=True)
        sim_cols[1].plotly_chart(px.line(skew_df, x="skew", y="final_pnl", markers=True, title=f"{selected_product} skew sweep"), use_container_width=True)
        if f"{selected_product}_bid_edge" in frame.columns or f"{selected_product}_ask_edge" in frame.columns:
            edge_trace = go.Figure()
            if f"{selected_product}_bid_edge" in frame.columns:
                edge_trace.add_trace(go.Scattergl(x=frame["timestamp"], y=frame[f"{selected_product}_bid_edge"], name="Bid edge", line=dict(color="#2563eb", width=1.6)))
            if f"{selected_product}_ask_edge" in frame.columns:
                edge_trace.add_trace(go.Scattergl(x=frame["timestamp"], y=frame[f"{selected_product}_ask_edge"], name="Ask edge", line=dict(color="#dc2626", width=1.6)))
            edge_trace.add_vline(x=selected_ts, line_width=1, line_dash="dash", line_color="#ef4444")
            edge_trace.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20), title="Logged quote edges", xaxis_title="Timestamp", yaxis_title="Edge")
            st.plotly_chart(edge_trace, use_container_width=True)


if __name__ == "__main__":
    main()
