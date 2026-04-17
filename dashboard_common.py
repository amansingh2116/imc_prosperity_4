from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as norm_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover - optional dependency fallback
    adfuller = None


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80,
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


@dataclass
class LogBundle:
    payload: dict[str, Any]
    market: pd.DataFrame
    trades: pd.DataFrame
    our_trades: pd.DataFrame
    logs: pd.DataFrame
    parsed_logs: pd.DataFrame
    graph: pd.DataFrame
    product_frames: dict[str, pd.DataFrame]
    products: list[str]
    total_profit: float


def get_limit(product: str, position_limits: dict[str, int] | None = None) -> int:
    limits = position_limits or DEFAULT_POSITION_LIMITS
    return int(limits.get(product, 80))


def discover_local_files(patterns: Sequence[str], root: Path | None = None) -> list[Path]:
    base = root or ROOT_DIR
    found: set[Path] = set()
    for pattern in patterns:
        found.update(base.glob(pattern))
    return sorted(path for path in found if path.is_file())


def source_name(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return getattr(source, "name", "uploaded")


def read_text_source(source: Any) -> str:
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        raise TypeError(f"Unsupported source type: {type(source)!r}")

    if isinstance(raw, str):
        return raw
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_table_source(source: Any) -> pd.DataFrame:
    text = read_text_source(source)
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    sep = ";" if first_line.count(";") >= first_line.count(",") else ","
    frame = pd.read_csv(StringIO(text), sep=sep)
    frame = frame.replace("", np.nan)
    return frame


def _numeric_columns(frame: pd.DataFrame, ignore: Iterable[str] = ()) -> pd.DataFrame:
    out = frame.copy()
    ignore_set = set(ignore)
    for column in out.columns:
        if column in ignore_set:
            continue
        try:
            out[column] = pd.to_numeric(out[column], errors="raise")
        except Exception:
            pass
    return out


def enrich_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    for column in (
        "bid_price_1",
        "bid_price_2",
        "bid_price_3",
        "ask_price_1",
        "ask_price_2",
        "ask_price_3",
        "bid_volume_1",
        "bid_volume_2",
        "bid_volume_3",
        "ask_volume_1",
        "ask_volume_2",
        "ask_volume_3",
        "mid_price",
        "profit_and_loss",
    ):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "mid_price" not in out.columns and {"bid_price_1", "ask_price_1"}.issubset(out.columns):
        out["mid_price"] = (out["bid_price_1"] + out["ask_price_1"]) / 2

    if "profit_and_loss" not in out.columns:
        out["profit_and_loss"] = 0.0

    bid_depth_cols = [col for col in ("bid_volume_1", "bid_volume_2", "bid_volume_3") if col in out.columns]
    ask_depth_cols = [col for col in ("ask_volume_1", "ask_volume_2", "ask_volume_3") if col in out.columns]

    out["spread"] = out.get("ask_price_1", np.nan) - out.get("bid_price_1", np.nan)
    out["bid_depth"] = out[bid_depth_cols].fillna(0).sum(axis=1) if bid_depth_cols else 0.0
    out["ask_depth"] = out[ask_depth_cols].fillna(0).sum(axis=1) if ask_depth_cols else 0.0
    out["arith_mid"] = (out.get("bid_price_1", np.nan) + out.get("ask_price_1", np.nan)) / 2

    best_bid_vol = out.get("bid_volume_1", pd.Series(0, index=out.index)).fillna(0)
    best_ask_vol = out.get("ask_volume_1", pd.Series(0, index=out.index)).fillna(0)
    denom = best_bid_vol + best_ask_vol + 1e-9
    out["wall_mid"] = (
        out.get("ask_price_1", np.nan) * best_bid_vol
        + out.get("bid_price_1", np.nan) * best_ask_vol
    ) / denom
    out["micro_price"] = out["wall_mid"]
    out["imbalance"] = (out["bid_depth"] - out["ask_depth"]) / (out["bid_depth"] + out["ask_depth"] + 1e-9)
    out["mid_change"] = out["mid_price"].diff()
    out["mid_return"] = out["mid_price"].pct_change()
    out["rolling_mean_200"] = out["mid_price"].rolling(200, min_periods=1).mean()
    out["rolling_vol_100"] = out["mid_return"].rolling(100, min_periods=10).std() * math.sqrt(100)

    if "day" in out.columns:
        out["intraday_demeaned"] = out.groupby("day")["mid_price"].transform(lambda s: s - s.mean())
    else:
        out["intraday_demeaned"] = out["mid_price"] - out["mid_price"].mean()

    return out


def load_price_data(sources: Sequence[Any]) -> pd.DataFrame:
    if not sources:
        return pd.DataFrame()

    frames = []
    for source in sources:
        frame = read_table_source(source)
        frame = _numeric_columns(frame, ignore=("product",))
        frame["__source__"] = source_name(source)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if "product" not in combined.columns and "symbol" in combined.columns:
        combined["product"] = combined["symbol"]
    if "day" not in combined.columns:
        combined["day"] = 0
    combined["day"] = pd.to_numeric(combined["day"], errors="coerce").fillna(0).astype(int)
    combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce")
    day_rank = {day: idx for idx, day in enumerate(sorted(combined["day"].dropna().unique()))}
    combined["global_time"] = combined["day"].map(day_rank).fillna(0).astype(int) * 1_000_000 + combined["timestamp"]
    combined = combined.sort_values(["product", "global_time"]).reset_index(drop=True)
    return enrich_price_frame(combined)


def load_trade_data(sources: Sequence[Any]) -> pd.DataFrame:
    if not sources:
        return pd.DataFrame()

    frames = []
    for source in sources:
        frame = read_table_source(source)
        frame = _numeric_columns(frame, ignore=("buyer", "seller", "symbol", "product", "currency"))
        frame["__source__"] = source_name(source)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if "symbol" not in combined.columns and "product" in combined.columns:
        combined["symbol"] = combined["product"]
    if "product" not in combined.columns and "symbol" in combined.columns:
        combined["product"] = combined["symbol"]
    return combined.replace("", np.nan)


def build_product_frames(
    prices_df: pd.DataFrame,
    parsed_logs: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if prices_df.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    products = prices_df["product"].dropna().astype(str).unique().tolist()
    for product in products:
        product_frame = enrich_price_frame(
            prices_df[prices_df["product"] == product].sort_values("global_time").reset_index(drop=True)
        )
        if parsed_logs is not None and not parsed_logs.empty:
            merge_cols = [col for col in parsed_logs.columns if col == "timestamp" or col.startswith(f"{product}_")]
            if merge_cols:
                product_frame = product_frame.merge(
                    parsed_logs[merge_cols].drop_duplicates(subset=["timestamp"]),
                    on="timestamp",
                    how="left",
                )
        frames[product] = product_frame
    return frames


def summarize_products(
    prices_df: pd.DataFrame,
    position_limits: dict[str, int] | None = None,
) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame()

    rows = []
    for product, frame in build_product_frames(prices_df).items():
        rows.append(
            {
                "product": product,
                "rows": len(frame),
                "days": frame["day"].nunique() if "day" in frame.columns else 1,
                "mid_mean": frame["mid_price"].mean(),
                "mid_std": frame["mid_price"].std(),
                "mid_min": frame["mid_price"].min(),
                "mid_max": frame["mid_price"].max(),
                "spread_mean": frame["spread"].mean(),
                "spread_std": frame["spread"].std(),
                "spread_min": frame["spread"].min(),
                "spread_max": frame["spread"].max(),
                "position_limit": get_limit(product, position_limits),
            }
        )
    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def fair_value_summary(frame: pd.DataFrame) -> dict[str, float]:
    simple_mean = float(frame["mid_price"].mean())
    total_depth = frame[
        [col for col in ("bid_volume_1", "bid_volume_2", "bid_volume_3", "ask_volume_1", "ask_volume_2", "ask_volume_3") if col in frame.columns]
    ].fillna(0).sum(axis=1)
    volume_weighted = float((frame["mid_price"] * total_depth).sum() / (total_depth.sum() + 1e-9))
    wall_mid = float(frame["wall_mid"].mean())
    rounded_mode = frame["mid_price"].round(0).mode()
    mode_value = float(rounded_mode.iloc[0]) if not rounded_mode.empty else simple_mean
    return {
        "Simple Mean": simple_mean,
        "Volume-Weighted Mean": volume_weighted,
        "Wall Mid": wall_mid,
        "Mode": mode_value,
    }


def shapiro_test_result(series: pd.Series, sample_size: int = 5000) -> dict[str, float | str]:
    values = series.dropna()
    if values.empty:
        return {"stat": np.nan, "p_value": np.nan, "verdict": "No data"}
    sample = values.sample(min(sample_size, len(values)), random_state=42)
    stat, p_value = stats.shapiro(sample)
    verdict = "Normal-ish" if p_value > 0.05 else "Not normal"
    return {"stat": float(stat), "p_value": float(p_value), "verdict": verdict}


def adf_test_result(series: pd.Series) -> dict[str, float | str]:
    values = series.dropna()
    if len(values) < 25:
        return {"stat": np.nan, "p_value": np.nan, "verdict": "Insufficient data"}
    if adfuller is None:
        return {"stat": np.nan, "p_value": np.nan, "verdict": "statsmodels unavailable"}
    stat, p_value, *_ = adfuller(values.values, maxlag=20, autolag="AIC")
    verdict = "Stationary" if p_value < 0.05 else "Non-stationary"
    return {"stat": float(stat), "p_value": float(p_value), "verdict": verdict}


def hurst_exponent(series: Sequence[float], max_lag: int = 100) -> float:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 100:
        return float("nan")
    lags = range(2, min(max_lag, len(values) // 2))
    tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    valid = [(lag, t) for lag, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return float("nan")
    lag_vals = np.log([lag for lag, _ in valid])
    tau_vals = np.log([t for _, t in valid])
    slope, _ = np.polyfit(lag_vals, tau_vals, 1)
    return float(slope)


def mean_reversion_halflife(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) < 10:
        return float("nan")
    y = values.diff().dropna().values
    x = values.shift(1).dropna().values
    beta = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
    if beta >= 0:
        return float("inf")
    return float(-math.log(2) / beta)


def acf_values(series: pd.Series, max_lag: int = 20) -> list[float]:
    values = series.dropna()
    return [float(values.autocorr(lag=lag)) for lag in range(1, max_lag + 1)]


def pacf_values(series: pd.Series, max_lag: int = 20) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    pacf: list[float] = []
    for lag in range(1, max_lag + 1):
        if len(values) <= lag + 2:
            pacf.append(float("nan"))
            continue
        design = np.column_stack([values[lag - step : -step] for step in range(1, lag + 1)])
        target = values[lag:]
        if design.size == 0 or target.size == 0:
            pacf.append(float("nan"))
            continue
        valid_rows = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
        if valid_rows.sum() <= lag + 1:
            pacf.append(float("nan"))
            continue
        design = design[valid_rows]
        target = target[valid_rows]
        if np.all(np.std(design, axis=0) < 1e-12) or np.std(target) < 1e-12:
            pacf.append(float("nan"))
            continue
        try:
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            pacf.append(float(coef[-1]))
        except np.linalg.LinAlgError:
            pacf.append(float("nan"))
    return pacf


def ar1_forecast(series: pd.Series, train_frac: float = 0.8) -> dict[str, float]:
    values = series.dropna().reset_index(drop=True).astype(float)
    if len(values) < 20:
        return {
            "c": np.nan,
            "phi": np.nan,
            "rmse": np.nan,
            "naive_rmse": np.nan,
            "r2_vs_naive": np.nan,
        }
    n = len(values)
    split = max(2, int(n * train_frac))
    y = values.values
    design = np.column_stack([np.ones(n - 1), y[:-1]])
    target = y[1:]
    coef = np.linalg.lstsq(design[: split - 1], target[: split - 1], rcond=None)[0]
    c, phi = coef
    prediction = c + phi * y[split:-1]
    actual = y[split + 1 :]
    rmse = float(np.sqrt(np.mean((prediction - actual) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((y[split:-1] - actual) ** 2)))
    r2 = 1 - (rmse**2) / (naive_rmse**2 + 1e-12)
    return {
        "c": float(c),
        "phi": float(phi),
        "rmse": rmse,
        "naive_rmse": naive_rmse,
        "r2_vs_naive": float(r2),
    }


def build_ml_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ret1"] = data["mid_price"].pct_change(1)
    data["ret2"] = data["mid_price"].pct_change(2)
    data["ret5"] = data["mid_price"].pct_change(5)
    data["ret10"] = data["mid_price"].pct_change(10)
    data["spread_pct"] = data["spread"] / (data["mid_price"] + 1e-9)
    data["dev_from_ema"] = data["mid_price"] - data["mid_price"].ewm(span=50, adjust=False).mean()
    data["vol_20"] = data["ret1"].rolling(20).std()
    data["target"] = (data["mid_price"].shift(-1) > data["mid_price"]).astype(int)
    feature_cols = ["ret1", "ret2", "ret5", "ret10", "imbalance", "spread_pct", "dev_from_ema", "vol_20"]
    return data.dropna(subset=feature_cols + ["target"])


def direction_model_summary(frame: pd.DataFrame) -> dict[str, Any]:
    feature_cols = ["ret1", "ret2", "ret5", "ret10", "imbalance", "spread_pct", "dev_from_ema", "vol_20"]
    data = build_ml_features(frame)
    if len(data) < 200:
        return {"feature_cols": feature_cols, "baseline": np.nan, "logistic_acc": np.nan, "mlp_acc": np.nan, "coefficients": {}}

    x = data[feature_cols].values
    y = data["target"].values
    split = int(len(data) * 0.7)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=500)
    logistic.fit(x_train_s, y_train)
    logistic_acc = accuracy_score(y_test, logistic.predict(x_test_s))

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=250, random_state=42)
    mlp.fit(x_train_s, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(x_test_s))

    baseline = max(y_test.mean(), 1 - y_test.mean())
    coefficients = {
        feature: float(coef)
        for feature, coef in sorted(zip(feature_cols, logistic.coef_[0]), key=lambda item: -abs(item[1]))
    }
    return {
        "feature_cols": feature_cols,
        "baseline": float(baseline),
        "logistic_acc": float(logistic_acc),
        "mlp_acc": float(mlp_acc),
        "coefficients": coefficients,
    }


def pairs_analysis(prices_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if prices_df.empty:
        return pd.DataFrame(), []
    pivot = prices_df.pivot_table(index="global_time", columns="product", values="mid_price")
    results: list[dict[str, Any]] = []
    for left, right in itertools.combinations(pivot.columns.tolist(), 2):
        s1 = pivot[left].dropna()
        s2 = pivot[right].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 100:
            continue
        x = s1.loc[common]
        y = s2.loc[common]
        beta = np.cov(x.values, y.values)[0, 1] / (np.var(y.values) + 1e-12)
        spread = x - beta * y
        roll_mean = spread.rolling(200, min_periods=50).mean()
        roll_std = spread.rolling(200, min_periods=50).std()
        last_z = float(((spread - roll_mean) / (roll_std + 1e-9)).iloc[-1])
        adf_result = adf_test_result(spread)
        results.append(
            {
                "pair": f"{left} vs {right}",
                "left": left,
                "right": right,
                "correlation": float(x.corr(y)),
                "beta": float(beta),
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std()),
                "spread_half_life": mean_reversion_halflife(spread),
                "spread_adf_p": adf_result["p_value"],
                "last_z_score": last_z,
            }
        )
    return pivot, results


def simulate_price_paths(
    series: pd.Series,
    n_paths: int = 500,
    n_steps: int = 250,
    method: str = "bootstrap",
    seed: int = 42,
) -> np.ndarray:
    values = series.dropna()
    if len(values) < 50:
        return np.empty((0, 0))
    returns = values.pct_change().dropna().values
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = values.iloc[-1]
    if method == "normal":
        simulated_returns = rng.normal(returns.mean(), returns.std(), size=(n_paths, n_steps))
    else:
        simulated_returns = rng.choice(returns, size=(n_paths, n_steps), replace=True)
    for step in range(n_steps):
        paths[:, step + 1] = paths[:, step] * (1 + simulated_returns[:, step])
    return paths


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, S - K)
        return {"price": intrinsic, "delta": 1.0 if S > K else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return {
        "price": float(S * norm_dist.cdf(d1) - K * math.exp(-r * T) * norm_dist.cdf(d2)),
        "delta": float(norm_dist.cdf(d1)),
        "gamma": float(norm_dist.pdf(d1) / (S * sigma * math.sqrt(T))),
        "vega": float(S * norm_dist.pdf(d1) * math.sqrt(T) / 100),
        "theta": float((-(S * norm_dist.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_dist.cdf(d2)) / 252),
    }


def implied_vol(market_price: float, S: float, K: float, T: float, r: float, tol: float = 1e-6) -> float:
    low, high = 0.001, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        price = black_scholes_call(S, K, T, r, mid)["price"]
        if abs(price - market_price) < tol:
            return float(mid)
        if price < market_price:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


def analyse_conversion_opportunity(
    local_bid: float,
    local_ask: float,
    foreign_bid: float,
    foreign_ask: float,
    transport_fees: float,
    import_tariff: float,
    export_tariff: float,
) -> dict[str, float]:
    import_cost = foreign_ask + transport_fees + import_tariff
    export_revenue = foreign_bid - transport_fees - export_tariff
    return {
        "import_cost": float(import_cost),
        "export_revenue": float(export_revenue),
        "import_arb": float(local_bid - import_cost),
        "export_arb": float(export_revenue - local_ask),
    }


def find_best_exchange_cycle(
    rates_dict: dict[tuple[str, str], float],
    start: str = "XIRECS",
    max_steps: int = 4,
) -> tuple[list[str], float]:
    best_path: list[str] = []
    best_profit = 1.0

    def dfs(current: str, path: list[str], product: float, steps: int) -> None:
        nonlocal best_path, best_profit
        if steps > 0 and current == start and product > best_profit:
            best_path = path[:]
            best_profit = product
            return
        if steps >= max_steps:
            return
        for (source, target), rate in rates_dict.items():
            if source != current:
                continue
            if target in path[1:] and target != start:
                continue
            dfs(target, path + [target], product * rate, steps + 1)

    dfs(start, [start], 1.0, 0)
    return best_path, float(best_profit)


def backtest_full_mm(
    frame: pd.DataFrame,
    trades_df: pd.DataFrame,
    product: str,
    fv_method: str = "fixed",
    fv_fixed: float | None = None,
    ema_alpha: float = 0.08,
    passive_edge: float = 4.0,
    aggr_edge: float = 1.0,
    skew_factor: float = 0.06,
    quote_size: int = 20,
    position_limit: int = 80,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    product_frame = frame.sort_values("global_time").reset_index(drop=True)
    product_trades = trades_df[trades_df.get("symbol", pd.Series(dtype=object)) == product].copy() if not trades_df.empty else pd.DataFrame()
    trade_groups = {timestamp: group for timestamp, group in product_trades.groupby("timestamp")} if not product_trades.empty else {}

    position = 0
    cash = 0.0
    ema_value: float | None = None
    rows: list[dict[str, float]] = []

    for row in product_frame.itertuples(index=False):
        mid = float(getattr(row, "mid_price"))
        best_bid = getattr(row, "bid_price_1", np.nan)
        best_ask = getattr(row, "ask_price_1", np.nan)
        bid_volume = getattr(row, "bid_volume_1", 0) or 0
        ask_volume = getattr(row, "ask_volume_1", 0) or 0

        if fv_method == "ema":
            ema_value = mid if ema_value is None else ema_alpha * mid + (1 - ema_alpha) * ema_value
            fair_value = float(ema_value)
        elif fv_method == "microprice":
            fair_value = float(getattr(row, "micro_price", mid))
        else:
            fair_value = float(fv_fixed if fv_fixed is not None else mid)

        skew = position * skew_factor
        bid_quote = round(fair_value - passive_edge - skew)
        ask_quote = round(fair_value + passive_edge - skew)
        if bid_quote >= ask_quote:
            bid_quote = math.floor(fair_value) - 1
            ask_quote = math.ceil(fair_value) + 1

        if pd.notna(best_ask) and float(best_ask) <= fair_value - aggr_edge and position < position_limit:
            quantity = min(int(ask_volume), quote_size, position_limit - position)
            if quantity > 0:
                position += quantity
                cash -= quantity * float(best_ask)

        if pd.notna(best_bid) and float(best_bid) >= fair_value + aggr_edge and position > -position_limit:
            quantity = min(int(bid_volume), quote_size, position_limit + position)
            if quantity > 0:
                position -= quantity
                cash += quantity * float(best_bid)

        if getattr(row, "timestamp") in trade_groups:
            for trade in trade_groups[getattr(row, "timestamp")].itertuples(index=False):
                trade_price = float(getattr(trade, "price"))
                quantity = int(getattr(trade, "quantity"))
                if trade_price <= bid_quote and position < position_limit:
                    filled = min(quantity, quote_size, position_limit - position)
                    if filled > 0:
                        position += filled
                        cash -= filled * bid_quote
                elif trade_price >= ask_quote and position > -position_limit:
                    filled = min(quantity, quote_size, position_limit + position)
                    if filled > 0:
                        position -= filled
                        cash += filled * ask_quote

        pnl = cash + position * mid
        rows.append(
            {
                "timestamp": float(getattr(row, "timestamp")),
                "global_time": float(getattr(row, "global_time")),
                "fv": fair_value,
                "bid_quote": float(bid_quote),
                "ask_quote": float(ask_quote),
                "position": float(position),
                "cash": float(cash),
                "pnl": float(pnl),
            }
        )

    return pd.DataFrame(rows)


def parse_activities_text(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    frame = pd.read_csv(StringIO(text), sep=";")
    frame = frame.replace("", np.nan)
    frame = _numeric_columns(frame, ignore=("product",))
    if "day" not in frame.columns:
        frame["day"] = 0
    frame["day"] = pd.to_numeric(frame["day"], errors="coerce").fillna(0).astype(int)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    day_rank = {day: idx for idx, day in enumerate(sorted(frame["day"].dropna().unique()))}
    frame["global_time"] = frame["day"].map(day_rank).fillna(0).astype(int) * 1_000_000 + frame["timestamp"]
    return enrich_price_frame(frame.sort_values(["product", "global_time"]).reset_index(drop=True))


def parse_graph_log_text(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(text), sep=";")


def parse_trade_history(raw_trades: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.DataFrame(raw_trades)
    if trades.empty:
        return trades, trades.copy()
    trades = trades.replace("", np.nan)
    trades = _numeric_columns(trades, ignore=("buyer", "seller", "symbol", "currency"))
    trades["is_buy_sub"] = trades["buyer"] == "SUBMISSION"
    trades["is_sell_sub"] = trades["seller"] == "SUBMISSION"
    trades["is_our_trade"] = trades["is_buy_sub"] | trades["is_sell_sub"]
    trades["side"] = np.where(trades["is_buy_sub"], "BUY", np.where(trades["is_sell_sub"], "SELL", "OTHER"))
    our_trades = trades[trades["is_our_trade"]].copy()
    return trades, our_trades


def _pick_best_value(values: list[Any]) -> Any:
    usable = [value for value in values if value not in (None, "", [], {})]
    if not usable:
        return values[-1] if values else None
    if all(isinstance(value, str) for value in usable):
        return max(usable, key=len)
    if all(isinstance(value, list) for value in usable):
        return max(usable, key=len)
    return usable[-1]


def merge_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys = set().union(*(payload.keys() for payload in payloads if payload))
    for key in keys:
        merged[key] = _pick_best_value([payload.get(key) for payload in payloads if payload])
    return merged


def extract_log_fields(logs_df: pd.DataFrame, products: Sequence[str]) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()

    records = []
    for row in logs_df.itertuples(index=False):
        lambda_log = getattr(row, "lambdaLog", "") or ""
        record: dict[str, Any] = {
            "timestamp": getattr(row, "timestamp", np.nan),
            "iteration": np.nan,
            "positions_text": np.nan,
        }
        lines = [line for line in str(lambda_log).splitlines() if line.strip()]
        iter_match = re.search(r"--- Iter\s+(\d+)\s+\|\s+t=(\d+)", lambda_log)
        if iter_match:
            record["iteration"] = float(iter_match.group(1))
        positions_line = next((line for line in lines if line.startswith("Positions:")), "")
        if positions_line:
            record["positions_text"] = positions_line.removeprefix("Positions:").strip()

        for product in products:
            product_line = next((line for line in lines if line.startswith(f"{product} |")), "")
            record[f"{product}_log_line"] = product_line or np.nan
            if not product_line:
                continue

            for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?\d+(?:\.\d+)?)", product_line):
                lower_key = key.lower()
                if lower_key == "edge":
                    continue
                record[f"{product}_{lower_key}"] = float(value)

            bid_edge_match = re.search(r"bid=(-?\d+(?:\.\d+)?)\(edge=([+-]?\d+(?:\.\d+)?)\)", product_line)
            ask_edge_match = re.search(r"ask=(-?\d+(?:\.\d+)?)\(edge=([+-]?\d+(?:\.\d+)?)\)", product_line)
            if bid_edge_match:
                record[f"{product}_bid"] = float(bid_edge_match.group(1))
                record[f"{product}_bid_edge"] = float(bid_edge_match.group(2))
            if ask_edge_match:
                record[f"{product}_ask"] = float(ask_edge_match.group(1))
                record[f"{product}_ask_edge"] = float(ask_edge_match.group(2))
        records.append(record)
    return pd.DataFrame(records)


def load_log_bundle(json_source: Any | None = None, log_source: Any | None = None) -> LogBundle:
    payloads = []
    for source in (json_source, log_source):
        if source is None:
            continue
        payloads.append(json.loads(read_text_source(source)))

    payload = merge_payloads(payloads) if payloads else {}
    market = parse_activities_text(payload.get("activitiesLog", ""))
    products = market["product"].dropna().astype(str).unique().tolist() if not market.empty else []
    trades, our_trades = parse_trade_history(payload.get("tradeHistory", []))
    logs = pd.DataFrame(payload.get("logs", []))
    graph = parse_graph_log_text(payload.get("graphLog", ""))
    parsed_logs = extract_log_fields(logs, products)
    product_frames = build_product_frames(market, parsed_logs)
    total_profit = float(payload.get("profit", 0.0) or 0.0)
    return LogBundle(
        payload=payload,
        market=market,
        trades=trades,
        our_trades=our_trades,
        logs=logs,
        parsed_logs=parsed_logs,
        graph=graph,
        product_frames=product_frames,
        products=products,
        total_profit=total_profit,
    )


def compute_leaderboard_metrics(
    market_df: pd.DataFrame,
    product_frames: dict[str, pd.DataFrame],
    our_trades: pd.DataFrame,
) -> dict[str, Any]:
    if market_df.empty or not product_frames:
        return {}

    timestamps = sorted(market_df["timestamp"].dropna().unique().tolist())
    pnl_df = pd.DataFrame({"timestamp": timestamps})
    for product, frame in product_frames.items():
        pnl_series = frame.set_index("timestamp")["profit_and_loss"]
        pnl_df[product] = pnl_df["timestamp"].map(pnl_series).ffill().fillna(0)
    pnl_df["total"] = pnl_df[[col for col in pnl_df.columns if col != "timestamp"]].sum(axis=1)

    total_pnl = float(pnl_df["total"].iloc[-1])
    pnl_array = pnl_df["total"].to_numpy()
    running_max = np.maximum.accumulate(pnl_array)
    drawdown = running_max - pnl_array
    max_drawdown = float(drawdown.max())
    trough_idx = int(drawdown.argmax())
    peak_idx = int(pnl_array[: trough_idx + 1].argmax()) if trough_idx > 0 else 0
    peak_ts = int(pnl_df["timestamp"].iloc[peak_idx])
    trough_ts = int(pnl_df["timestamp"].iloc[trough_idx])
    recovery_ts = None
    peak_value = pnl_array[peak_idx]
    for idx in range(trough_idx, len(pnl_array)):
        if pnl_array[idx] >= peak_value:
            recovery_ts = int(pnl_df["timestamp"].iloc[idx])
            break

    per_product: dict[str, dict[str, Any]] = {}
    for product, frame in product_frames.items():
        product_pnl = float(frame["profit_and_loss"].iloc[-1])
        product_array = frame["profit_and_loss"].ffill().fillna(0).to_numpy()
        product_running_max = np.maximum.accumulate(product_array)
        product_drawdown = float((product_running_max - product_array).max())
        fills = our_trades[our_trades["symbol"] == product] if not our_trades.empty else pd.DataFrame()
        per_product[product] = {
            "pnl": product_pnl,
            "max_drawdown": product_drawdown,
            "recovery_ratio": product_pnl / (product_drawdown + 1e-9),
            "fills": int(len(fills)),
            "avg_fill": float(fills["quantity"].mean()) if not fills.empty else 0.0,
            "pnl_share_pct": 100 * product_pnl / (total_pnl + 1e-9),
        }

    return {
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "recovery_ratio": total_pnl / (max_drawdown + 1e-9),
        "avg_fill": float(our_trades["quantity"].mean()) if not our_trades.empty else 0.0,
        "total_fills": int(len(our_trades)),
        "peak_ts": peak_ts,
        "trough_ts": trough_ts,
        "recovery_ts": recovery_ts,
        "timestamps": pnl_df["timestamp"].tolist(),
        "pnl_series": pnl_array,
        "drawdown_series": drawdown,
        "pnl_df": pnl_df,
        "per_product": per_product,
    }
