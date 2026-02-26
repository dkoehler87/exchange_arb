# -*- coding: utf-8 -*-
"""
Exchange Arbitrage Demo (Streamlit)

Updates applied per request:
1) OKX left uncolored; Gate.io dark blue.
2) Exchange-vs-exchange matrix:
   - Exchange header "cells" colored (row + column) reliably (implemented as a display matrix row/col, not Styler headers).
   - Selected pair shown in the top-left corner with light-blue background.
3) Section 2 layout cleanup:
   - Candidate path leaderboard first (top row highlighted light green) and includes Venue Path.
   - Select a candidate (dropdown) to drive the "Path legs" table.
   - Size impact vs direct hedge moved into its own section below.
   - Arbitrage path map full-width, large, with always-visible labels.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd
import streamlit as st


# ============================
# CONFIG
# ============================

DEFAULT_EXCHANGES = ["kraken", "coinbase", "bitstamp", "okx", "bitfinex", "gateio"]

STANDARD_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BTC/USD", "ETH/USD", "BTC/EUR", "ETH/EUR"]

STABLE_FX_SYMBOLS = ["USDT/USD", "USDT/EUR", "USDC/USD", "USDC/EUR", "EUR/USD", "GBP/USD", "CAD/USD", "AUD/USD", "USD/JPY"]

DEFAULT_STANDARD_PICK = ["BTC/USDT", "BTC/EUR", "BTC/USD", "ETH/USD", "ETH/EUR"]
DEFAULT_STABLEFX_PICK = ["EUR/USD", "GBP/USD", "USD/JPY", "USDT/USD", "USDT/EUR"]

FX_FEEDS = ["EUR/USD", "GBP/USD", "CAD/USD", "AUD/USD", "USD/JPY"]

REQUEST_TIMEOUT_MS = 10000
AUTO_REFRESH_MS = 5000

DEFAULT_TAKER_FEE_STANDARD = 0.001
DEFAULT_TAKER_FEE_STABLEFX = 0.001

EXCHANGE_TAKER_FEES_STANDARD = {
    "kraken": 0.0016,
    "coinbase": 0.0010,
    "bitstamp": 0.0016,
    "okx": 0.0010,
    "bitfinex": 0.0000,
    "gateio": 0.0010,
}
EXCHANGE_TAKER_FEES_STABLEFX = {
    "kraken": 0.0002,
    "coinbase": 0.0001,
    "bitstamp": 0.0016,
    "okx": 0.0010,
    "bitfinex": 0.0000,
    "gateio": 0.0010,
}


# ============================
# STYLE: EXCHANGE COLORS
# ============================

# OKX intentionally uncolored
EXCHANGE_COLOR_MAP = {
    "kraken":   {"bg": "#6f42c1", "fg": "#ffffff"},  # purple
    "bitfinex": {"bg": "#0b6b3a", "fg": "#ffffff"},  # dark green
    "coinbase": {"bg": "#1a73e8", "fg": "#ffffff"},  # blue
    "bitstamp": {"bg": "#88d498", "fg": "#0b1f12"},  # light green
    # okx intentionally left uncolored
    "gateio":   {"bg": "#0b3c6d", "fg": "#ffffff"},  # dark blue
}

PAIR_TL_STYLE = "background-color: #dbeafe; color: #0b1f46; font-weight: 700; text-align: center;"

def exch_style(ex: str) -> str:
    ex_l = (ex or "").lower()
    c = EXCHANGE_COLOR_MAP.get(ex_l)
    if not c:
        return ""
    return f"background-color: {c['bg']}; color: {c['fg']}; font-weight: 600;"


def style_exchange_cells(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return per-cell CSS for a set of columns that contain exchange names."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in columns:
        if col not in df.columns:
            continue
        for i in df.index:
            v = df.at[i, col]
            if pd.isna(v):
                continue
            styles.at[i, col] = exch_style(str(v))
    return styles


# ============================
# HEADER CLEANUP
# ============================

def prettify_headers(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).replace("_", " ").strip()
        out.append(" ".join([w.capitalize() if w.isalpha() else w for w in s.split(" ")]))
    return out


def prettify_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = prettify_headers(list(df2.columns))
    return df2


# ============================
# STREAMLIT HELPERS
# ============================

def df_show(data, *, width: str = "stretch", hide_index: bool = True, height: Optional[int] = None):
    """Streamlit dataframe wrapper with modern width API."""
    kwargs = {"width": width, "hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    try:
        return st.dataframe(data, **kwargs)
    except TypeError:
        # older streamlit
        kwargs.pop("width", None)
        kwargs["use_container_width"] = (width == "stretch")
        return st.dataframe(data, **kwargs)




def render_styler_html(styler, *, css_class: str, hide_thead: bool = False, hide_index: bool = True):
    """
    Render a pandas Styler via HTML so we can fully control headers (Streamlit's st.dataframe
    can render an extra native header row and an index column).
    """
    try:
        styler = styler.set_table_attributes(f'class="{css_class}"')
    except Exception:
        pass

    css = f"<style>.{css_class} {{width: 100%;}}"
    if hide_thead:
        css += f".{css_class} thead {{display:none;}}"
    if hide_index:
        # Hide the left-most index/row-number column produced by pandas Styler HTML
        css += f".{css_class} th.row_heading, .{css_class} th.index_name {{display:none;}}"
        css += f".{css_class} tbody th {{display:none;}}"
    css += "</style>"

    html = styler.to_html()
    st.markdown(css + html, unsafe_allow_html=True)


def parse_number_with_commas(s: str) -> Optional[float]:
    try:
        if s is None:
            return None
        t = str(s).strip()
        if t == "":
            return None
        t = t.replace(",", "").replace("_", "")
        return float(t)
    except Exception:
        return None


def fmt_int_commas(x: float) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return ""


# ============================
# MODELS / CORE HELPERS
# ============================

@dataclass
class TopOfBook:
    bid: float
    ask: float
    timestamp: Optional[int] = None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def mk_exchange(name: str) -> ccxt.Exchange:
    ex_class = getattr(ccxt, name)
    return ex_class({"enableRateLimit": True, "timeout": REQUEST_TIMEOUT_MS})


def fetch_tob(ex: ccxt.Exchange, symbol: str) -> Optional[TopOfBook]:
    try:
        if ex.id == "bitfinex":
            ob = ex.fetch_order_book(symbol, params={"len": 25})
        else:
            ob = ex.fetch_order_book(symbol, limit=5)
    except TypeError:
        ob = ex.fetch_order_book(symbol)

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return None

    bid = safe_float(bids[0][0])
    ask = safe_float(asks[0][0])
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None

    return TopOfBook(bid=bid, ask=ask, timestamp=ob.get("timestamp"))


def mid(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def parse_symbol(sym: str) -> Tuple[str, str]:
    base, quote = sym.split("/")
    return base.strip().upper(), quote.strip().upper()


def symbol_category(symbol: str) -> str:
    if symbol in STANDARD_SYMBOLS:
        return "standard"
    if symbol in STABLE_FX_SYMBOLS:
        return "stablefx"

    base, quote = parse_symbol(symbol)
    stable_like = {"USD", "EUR", "GBP", "JPY", "TRY", "USDT", "USDC", "CAD", "AUD"}
    if base in stable_like or quote in stable_like:
        return "stablefx"
    return "standard"


def get_taker_fee(ex_name: str, category: str) -> float:
    if category == "stablefx":
        return float(EXCHANGE_TAKER_FEES_STABLEFX.get(ex_name, DEFAULT_TAKER_FEE_STABLEFX))
    return float(EXCHANGE_TAKER_FEES_STANDARD.get(ex_name, DEFAULT_TAKER_FEE_STANDARD))


# ============================
# QUOTE FETCHING
# ============================

def fetch_symbol_quote_one_exchange(ex_name: str, symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        ex = mk_exchange(ex_name)
        ex.load_markets()
    except Exception as e:
        return None, {"exchange": ex_name, "symbol": symbol, "stage": "load_markets", "error": f"{type(e).__name__}: {e}"}

    if symbol not in ex.markets:
        return None, {"exchange": ex_name, "symbol": symbol, "stage": "symbol_missing", "error": "symbol not listed"}

    try:
        tob = fetch_tob(ex, symbol)
        if not tob:
            return None, {"exchange": ex_name, "symbol": symbol, "stage": "orderbook_empty", "error": "no bids/asks returned"}

        m = mid(tob.bid, tob.ask)
        return {
            "exchange": ex_name,
            "symbol": symbol,
            "bid": tob.bid,
            "ask": tob.ask,
            "mid": m,
            "timestamp_ms": tob.timestamp,
        }, None

    except Exception as e:
        return None, {"exchange": ex_name, "symbol": symbol, "stage": "fetch_order_book", "error": f"{type(e).__name__}: {e}"}


@st.cache_data(ttl=5, show_spinner=False)
def fetch_quotes_for_symbol(exchange_names: Tuple[str, ...], symbol: str, max_workers: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quotes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_symbol_quote_one_exchange, ex, symbol) for ex in exchange_names]
        for f in as_completed(futures):
            q, e = f.result()
            if q:
                quotes.append(q)
            if e:
                errors.append(e)

    return pd.DataFrame(quotes), pd.DataFrame(errors)


@st.cache_data(ttl=5, show_spinner=False)
def fetch_quotes_for_symbols_all_exchanges(
    exchange_names: Tuple[str, ...],
    symbols: Tuple[str, ...],
    max_workers: int = 18,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quotes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_symbol_quote_one_exchange, ex, sym) for ex in exchange_names for sym in symbols]
        for f in as_completed(futures):
            q, e = f.result()
            if q:
                quotes.append(q)
            if e:
                errors.append(e)

    return pd.DataFrame(quotes), pd.DataFrame(errors)


# ============================
# SECTION 1 MATRIX
# ============================

def build_exchange_matrix_values(df_quotes: pd.DataFrame, exchange_names: List[str], symbol: str) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Build raw matrix values (strings) and returns present exchanges + their mids."""
    present = []
    mids: Dict[str, float] = {}

    if df_quotes is not None and not df_quotes.empty:
        present_set = set(df_quotes["exchange"].astype(str).tolist())
        present = [ex for ex in exchange_names if ex in present_set]

        for ex in present:
            r = df_quotes[df_quotes["exchange"] == ex]
            if not r.empty:
                mids[ex] = float(r.iloc[0]["mid"])

    matrix = pd.DataFrame(index=present, columns=present, data="")

    for r_ex in present:
        for c_ex in present:
            if r_ex == c_ex:
                matrix.loc[r_ex, c_ex] = "—"
                continue

            r_mid = mids.get(r_ex)
            c_mid = mids.get(c_ex)
            if r_mid is None or c_mid is None or r_mid <= 0:
                matrix.loc[r_ex, c_ex] = ""
                continue

            delta = c_mid - r_mid
            bps = (delta / r_mid) / 0.0001
            cat = symbol_category(symbol)
            if cat == "stablefx":
                matrix.loc[r_ex, c_ex] = f"{delta:,.6f} | {bps:,.4f} bps"
            else:
                matrix.loc[r_ex, c_ex] = f"{delta:,.2f} | {bps:,.2f} bps"

    return matrix, present, mids


def matrix_display_table(matrix: pd.DataFrame, present: List[str], pair_label: str) -> pd.DataFrame:
    """
    Build a display-friendly matrix that *does not rely on Styler index/col headers*,
    because Streamlit sometimes drops header background styles.

    Layout:
      - Row 0 acts as header row: [pair_label, ex1, ex2, ...]
      - First column acts as row header: [ex1, ex2, ...]
    """
    if matrix is None or matrix.empty or not present:
        return pd.DataFrame()

    header = [""] + present
    rows = []
    rows.append([pair_label] + present)  # header row (top-left is pair label)

    for r_ex in present:
        row = [r_ex] + [matrix.loc[r_ex, c_ex] for c_ex in present]
        rows.append(row)

    disp = pd.DataFrame(rows, columns=header)
    return disp


def style_matrix_display(disp: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=disp.index, columns=disp.columns)

    # top-left pair label
    if disp.shape[0] > 0 and disp.shape[1] > 0:
        styles.iloc[0, 0] = PAIR_TL_STYLE

    # top header row exchange cells
    if disp.shape[0] > 0:
        for j in range(1, disp.shape[1]):
            styles.iloc[0, j] = exch_style(str(disp.iloc[0, j]))

    # first column exchange cells
    if disp.shape[1] > 0:
        for i in range(1, disp.shape[0]):
            styles.iloc[i, 0] = exch_style(str(disp.iloc[i, 0]))

    # data cell coloring (delta sign)
    for i in range(1, disp.shape[0]):
        for j in range(1, disp.shape[1]):
            v = disp.iloc[i, j]
            if v in ("—", "", None) or (isinstance(v, float) and pd.isna(v)):
                styles.iloc[i, j] = "color: #888;"
                continue

            try:
                delta_str = str(v).split("|")[0].strip().replace(",", "")
                delta = float(delta_str)
                if delta > 0:
                    styles.iloc[i, j] = "color: #0b6; font-weight: 600;"
                elif delta < 0:
                    styles.iloc[i, j] = "color: #d33; font-weight: 600;"
                else:
                    styles.iloc[i, j] = "color: #e5e7eb; font-weight: 600;"
            except Exception:
                styles.iloc[i, j] = "color: #e5e7eb;"

    return styles


# ============================
# TOP: FX FEEDS (compact)
# ============================

def summarize_fx(df_fx_quotes: pd.DataFrame) -> pd.DataFrame:
    if df_fx_quotes.empty:
        return pd.DataFrame(columns=["pair", "best_bid", "bid_ex", "best_ask", "ask_ex", "mid", "spread_bps", "venues"])

    rows = []
    for sym, g in df_fx_quotes.groupby("symbol"):
        g2 = g.copy()
        g2["bid"] = g2["bid"].astype(float)
        g2["ask"] = g2["ask"].astype(float)

        bid_row = g2.loc[g2["bid"].idxmax()] if not g2.empty else None
        ask_row = g2.loc[g2["ask"].idxmin()] if not g2.empty else None

        best_bid = float(bid_row["bid"]) if bid_row is not None else None
        bid_ex = str(bid_row["exchange"]) if bid_row is not None else None
        best_ask = float(ask_row["ask"]) if ask_row is not None else None
        ask_ex = str(ask_row["exchange"]) if ask_row is not None else None

        mid_val = None
        spread_bps = None
        if best_bid and best_ask and best_bid > 0 and best_ask > 0:
            mid_val = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid_val) / 0.0001

        rows.append({
            "pair": sym,
            "best_bid": best_bid,
            "bid_ex": bid_ex,
            "best_ask": best_ask,
            "ask_ex": ask_ex,
            "mid": mid_val,
            "spread_bps": spread_bps,
            "venues": int(len(g2)),
        })

    out = pd.DataFrame(rows).sort_values("pair")
    return out


@st.cache_data(ttl=5, show_spinner=False)
def fetch_fx_feeds(exchange_names: Tuple[str, ...], pairs: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quotes: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch_symbol_quote_one_exchange, ex, p) for ex in exchange_names for p in pairs]
        for f in as_completed(futures):
            q, e = f.result()
            if q:
                quotes.append(q)
            if e:
                errors.append(e)

    return pd.DataFrame(quotes), pd.DataFrame(errors)


# ============================
# SECTION 2: CROSS-EXCHANGE MULTI-LEG
# ============================

@dataclass
class XEdge:
    src: str
    dst: str
    rate: float          # dst per 1 src (net of fees)
    symbol: str
    action: str
    price_used: float
    fee_bps: float
    exchange: str


def build_best_cross_exchange_edges(df_quotes: pd.DataFrame, apply_fees: bool) -> List[XEdge]:
    if df_quotes.empty:
        return []

    edges: List[XEdge] = []

    for symbol, g in df_quotes.groupby("symbol"):
        base, quote = parse_symbol(symbol)
        cat = symbol_category(symbol)

        # QUOTE -> BASE (buy base @ ask): maximize rate = (1/ask)*(1-fee)
        best_buy = None
        best_buy_rate = -1.0
        for _, r in g.iterrows():
            ex = str(r["exchange"])
            ask = float(r["ask"])
            if ask <= 0:
                continue
            fee = get_taker_fee(ex, cat) if apply_fees else 0.0
            rate = (1.0 / ask) * (1.0 - fee)
            if rate > best_buy_rate:
                best_buy_rate = rate
                best_buy = (ex, ask, fee)

        if best_buy is not None and best_buy_rate > 0:
            ex, ask, fee = best_buy
            edges.append(XEdge(
                src=quote,
                dst=base,
                rate=best_buy_rate,
                symbol=symbol,
                action=f"BUY {base}",
                price_used=float(ask),
                fee_bps=float(fee) * 10000,
                exchange=ex,
            ))

        # BASE -> QUOTE (sell base @ bid): maximize rate = bid*(1-fee)
        best_sell = None
        best_sell_rate = -1.0
        for _, r in g.iterrows():
            ex = str(r["exchange"])
            bid = float(r["bid"])
            if bid <= 0:
                continue
            fee = get_taker_fee(ex, cat) if apply_fees else 0.0
            rate = bid * (1.0 - fee)
            if rate > best_sell_rate:
                best_sell_rate = rate
                best_sell = (ex, bid, fee)

        if best_sell is not None and best_sell_rate > 0:
            ex, bid, fee = best_sell
            edges.append(XEdge(
                src=base,
                dst=quote,
                rate=best_sell_rate,
                symbol=symbol,
                action=f"SELL {base}",
                price_used=float(bid),
                fee_bps=float(fee) * 10000,
                exchange=ex,
            ))

    return edges


def _keep_best(cands: List[Tuple[float, List[XEdge]]], k: int) -> List[Tuple[float, List[XEdge]]]:
    cands = sorted(cands, key=lambda x: x[0], reverse=True)
    seen = set()
    out = []
    for rate, path in cands:
        sig = tuple((pe.symbol, pe.src, pe.dst, pe.exchange) for pe in path)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((rate, path))
        if len(out) >= k:
            break
    return out


def enumerate_candidate_paths(
    edges: List[XEdge],
    start: str,
    end: str,
    max_hops: int,
    per_node_k: int = 10,
    top_results: int = 50,
) -> List[Tuple[int, float, List[XEdge]]]:
    """Return list of (hops, rate, path_edges) to end, pruned DP."""
    start = start.upper()
    end = end.upper()

    outgoing: Dict[str, List[XEdge]] = {}
    for e in edges:
        outgoing.setdefault(e.src, []).append(e)

    dp: List[Dict[str, List[Tuple[float, List[XEdge]]]]] = []
    dp.append({start: [(1.0, [])]})

    for h in range(1, max_hops + 1):
        prev = dp[h - 1]
        cur: Dict[str, List[Tuple[float, List[XEdge]]]] = {}

        for node, paths in prev.items():
            for rate, path in paths:
                for e in outgoing.get(node, []):
                    # prevent repeating the same (symbol, exchange) within a single path
                    if any((pe.symbol == e.symbol and pe.exchange == e.exchange) for pe in path):
                        continue
                    new_rate = rate * e.rate
                    new_path = path + [e]
                    cur.setdefault(e.dst, []).append((new_rate, new_path))

        for n in list(cur.keys()):
            cur[n] = _keep_best(cur[n], per_node_k)

        dp.append(cur)

    candidates: List[Tuple[int, float, List[XEdge]]] = []
    for h in range(1, max_hops + 1):
        for rate, path in dp[h].get(end, []):
            candidates.append((h, rate, path))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_results]
    return candidates


def best_paths_max_hops_cross(edges: List[XEdge], start: str, end: str, max_hops: int, top_k: int = 2) -> List[Tuple[float, List[XEdge]]]:
    cands = enumerate_candidate_paths(edges, start, end, max_hops=max_hops, per_node_k=10, top_results=max(25, top_k * 5))
    # unique bests
    out: List[Tuple[float, List[XEdge]]] = []
    seen = set()
    for hops, rate, path in sorted(cands, key=lambda x: x[1], reverse=True):
        sig = tuple((pe.symbol, pe.src, pe.dst, pe.exchange) for pe in path)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((rate, path))
        if len(out) >= top_k:
            break
    return out


def xedges_to_table(path: List[XEdge]) -> pd.DataFrame:
    rows = []
    for i, e in enumerate(path, start=1):
        rows.append({
            "#": i,
            "from": e.src,
            "to": e.dst,
            "symbol": e.symbol,
            "exchange": e.exchange,
            "action": e.action,
            "price_used": e.price_used,
            "fee_bps": e.fee_bps,
            "to_per_1_from (net)": e.rate,
        })
    return pd.DataFrame(rows)


def path_asset_text(start: str, path: List[XEdge]) -> str:
    return " → ".join([start.upper()] + [e.dst for e in path])


def path_venue_text(path: List[XEdge]) -> str:
    return " → ".join([e.exchange for e in path])


def build_leaderboard(candidates: List[Tuple[int, float, List[XEdge]]], start: str) -> Tuple[pd.DataFrame, List[List[XEdge]]]:
    rows = []
    paths: List[List[XEdge]] = []
    for i, (hops, rate, path) in enumerate(candidates, start=1):
        rows.append({
            "Rank": i,
            "Hops": hops,
            "Rate (End per 1 Start)": float(rate),
            "Asset Path": path_asset_text(start, path),
            "Venue Path": path_venue_text(path),
        })
        paths.append(path)
    df = pd.DataFrame(rows)
    return df, paths


def style_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if not df.empty:
        styles.iloc[0, :] = "background-color: #bbf7d0; color: #052e16; font-weight: 700;"  # readable highlight for bestn highlight for best
    return styles


def build_path_map_points_from_paths(
    start: str,
    candidates: List[Tuple[int, float, List[XEdge]]],
    best_path: Optional[List[XEdge]] = None,
) -> pd.DataFrame:
    """
    Build a point table for the path map.
    IMPORTANT: We mark the best path by matching the actual best_path legs (exchange+symbol+src+dst),
    not by assuming it's the first candidate row.
    """
    best_sig = None
    if best_path:
        best_sig = tuple((e.exchange, e.symbol, e.src, e.dst) for e in best_path)

    rows = []
    for i, (hops, rate, path) in enumerate(candidates, start=1):
        pid = f"P{i:02d}"
        sig = tuple((e.exchange, e.symbol, e.src, e.dst) for e in path)
        is_best = (best_sig is not None and sig == best_sig)

        # step 0 node (start) has no exchange
        rows.append({
            "path_id": pid,
            "step": 0,
            "node": start.upper(),
            "exchange": "",
            "label": start.upper(),
            "rate": float(rate),
            "hops": int(hops),
            "is_best": is_best,
            "asset_path": path_asset_text(start, path),
            "venue_path": path_venue_text(path),
        })

        # subsequent nodes from edges
        for s, e in enumerate(path, start=1):
            node = e.dst
            ex = e.exchange
            rows.append({
                "path_id": pid,
                "step": s,
                "node": node,
                "exchange": ex,
                "label": f"{node} @ {ex}",
                "rate": float(rate),
                "hops": int(hops),
                "is_best": is_best,
                "asset_path": path_asset_text(start, path),
                "venue_path": path_venue_text(path),
            })

    # Fallback: if best_path wasn't found among candidates (due to pruning), highlight candidate #1.
    df = pd.DataFrame(rows)
    if not df.empty and df["is_best"].sum() == 0:
        first_pid = df["path_id"].min()
        df.loc[df["path_id"] == first_pid, "is_best"] = True

    return df

def build_rate_progress_points_from_candidates(
    start: str,
    candidates: List[Tuple[int, float, List[XEdge]]],
    best_path: Optional[List[XEdge]] = None,
) -> pd.DataFrame:
    """
    Build points for a 'rate progression' chart.

    x-axis: step (0..hops)
    y-axis: cumulative rate after each leg (end units per 1 start unit).

    The best path is identified by matching the actual best_path signature
    (exchange, symbol, src, dst) – not by assuming candidate rank #1.
    """
    start = start.upper()

    best_sig = None
    if best_path:
        best_sig = tuple((e.exchange, e.symbol, e.src, e.dst) for e in best_path)

    rows: List[Dict[str, Any]] = []

    for i, (hops, final_rate, path) in enumerate(candidates, start=1):
        pid = f"P{i:02d}"
        sig = tuple((e.exchange, e.symbol, e.src, e.dst) for e in path)
        is_best = (best_sig is not None and sig == best_sig)

        asset_path = path_asset_text(start, path)
        venue_path = path_venue_text(path)

        # Use the *actual* per-leg edge rates for the y-axis (so endpoints are truly distinct)
        # rather than relying on any cached/rounded candidate final_rate.
        cum = 1.0
        final_rate_calc = 1.0
        for e in path:
            final_rate_calc *= float(e.rate)
        rows.append({
            "path_id": pid,
            "step": 0,
            "cum_rate": 1.0,
            "log_cum_rate": 0.0,
            "is_best": is_best,
            "is_final": (hops == 0),
            "final_rate": float(final_rate_calc),
            "asset_path": asset_path,
            "venue_path": venue_path,
            "leg_symbol": "",
            "leg_exchange": "",
            "leg_action": "",
        })

        for s, e in enumerate(path, start=1):
            cum *= float(e.rate)
            rows.append({
                "path_id": pid,
                "step": s,
                "cum_rate": float(cum),
                "log_cum_rate": float(math.log(cum)) if cum > 0 else None,
                "is_best": is_best,
                "is_final": (s == int(hops)),
                "final_rate": float(final_rate_calc),
                "asset_path": asset_path,
                "venue_path": venue_path,
                "leg_symbol": e.symbol,
                "leg_exchange": e.exchange,
                "leg_action": e.action,
            })

    df = pd.DataFrame(rows)

    # Fallback: if best_path wasn't found among candidates (e.g., pruning), highlight candidate #1.
    if not df.empty and df["is_best"].sum() == 0:
        first_pid = df["path_id"].min()
        df.loc[df["path_id"] == first_pid, "is_best"] = True

    return df



def best_direct_rate_start_to_end(df_quotes: pd.DataFrame, start_ccy: str, end_ccy: str, apply_fees: bool) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if df_quotes.empty:
        return None, None, None

    start_ccy = start_ccy.upper()
    end_ccy = end_ccy.upper()

    sym_sell = f"{start_ccy}/{end_ccy}"
    sym_buy = f"{end_ccy}/{start_ccy}"

    best_rate = None
    best_symbol = None
    best_exchange = None

    # SELL start for end using start/end
    g = df_quotes[df_quotes["symbol"] == sym_sell]
    if not g.empty:
        for _, r in g.iterrows():
            ex = str(r["exchange"])
            bid = float(r["bid"])
            if bid <= 0:
                continue
            fee = get_taker_fee(ex, symbol_category(sym_sell)) if apply_fees else 0.0
            rate = bid * (1.0 - fee)
            if best_rate is None or rate > best_rate:
                best_rate, best_symbol, best_exchange = rate, sym_sell, ex

    # BUY end using start via end/start
    g = df_quotes[df_quotes["symbol"] == sym_buy]
    if not g.empty:
        for _, r in g.iterrows():
            ex = str(r["exchange"])
            ask = float(r["ask"])
            if ask <= 0:
                continue
            fee = get_taker_fee(ex, symbol_category(sym_buy)) if apply_fees else 0.0
            rate = (1.0 / ask) * (1.0 - fee)
            if best_rate is None or rate > best_rate:
                best_rate, best_symbol, best_exchange = rate, sym_buy, ex

    return best_rate, best_symbol, best_exchange


# ============================
# UI
# ============================

st.set_page_config(page_title="Exchange Arb Demo", layout="wide")
st.title("Exchange Arbitrage Demo")

# ---- Sidebar ----
with st.sidebar:
    st.header("Controls")

    exchange_names = st.multiselect("Exchanges", options=DEFAULT_EXCHANGES, default=DEFAULT_EXCHANGES)

    st.divider()
    st.subheader("Symbols")
    standard_pick = st.multiselect("Standard symbols", options=STANDARD_SYMBOLS, default=[s for s in DEFAULT_STANDARD_PICK if s in STANDARD_SYMBOLS])
    stablefx_pick = st.multiselect("Stable & FX symbols", options=STABLE_FX_SYMBOLS, default=[s for s in DEFAULT_STABLEFX_PICK if s in STABLE_FX_SYMBOLS])

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)
    refresh_now = st.button("Refresh top panels now")
    clear_cache = st.button("Clear cache")

if clear_cache:
    st.cache_data.clear()
    for k in ["top_fx_key", "top_fx_summary", "top_fx_errors", "top_prices_key", "top_prices_quotes", "top_prices_errors", "matrix_symbol"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

if not exchange_names:
    st.warning("Select at least one exchange.")
    st.stop()

# ---- Autorefresh tick ----
tick = 0
if auto_refresh and not refresh_now:
    if hasattr(st, "autorefresh"):
        tick = st.autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh")
    elif hasattr(st, "st_autorefresh"):
        tick = st.st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh")

# ============================
# Universe / persistent pair
# ============================

matrix_universe = (standard_pick + stablefx_pick) if (standard_pick or stablefx_pick) else (STANDARD_SYMBOLS + STABLE_FX_SYMBOLS)
if "matrix_symbol" not in st.session_state:
    st.session_state["matrix_symbol"] = matrix_universe[0] if matrix_universe else "BTC/USD"

pair_for_top = st.session_state["matrix_symbol"]

# ============================
# TOP PANELS
# ============================

fx_key = (tuple(exchange_names), tick, bool(refresh_now))
prices_key = (tuple(exchange_names), pair_for_top, tick, bool(refresh_now))

if st.session_state.get("top_fx_key") != fx_key:
    with st.spinner("Updating FX feeds..."):
        df_fx_quotes, df_fx_errors = fetch_fx_feeds(tuple(exchange_names), tuple(FX_FEEDS))
        st.session_state["top_fx_summary"] = summarize_fx(df_fx_quotes)
        st.session_state["top_fx_errors"] = df_fx_errors
        st.session_state["top_fx_key"] = fx_key

if st.session_state.get("top_prices_key") != prices_key:
    with st.spinner("Updating current live prices..."):
        df_pq, df_pe = fetch_quotes_for_symbol(tuple(exchange_names), pair_for_top)
        st.session_state["top_prices_quotes"] = df_pq
        st.session_state["top_prices_errors"] = df_pe
        st.session_state["top_prices_key"] = prices_key

df_fx_summary = st.session_state.get("top_fx_summary", pd.DataFrame())
df_sym_quotes_top = st.session_state.get("top_prices_quotes", pd.DataFrame())

# ============================
# TOP SUMMARY ROW
# ============================

top_cols = st.columns([1.0, 1.4, 1.4], gap="large")

with top_cols[0]:
    st.subheader("Taker Fee Assumptions")
    df_fees = pd.DataFrame({
        "exchange": exchange_names,
        "standard_taker_fee": [EXCHANGE_TAKER_FEES_STANDARD.get(ex, DEFAULT_TAKER_FEE_STANDARD) for ex in exchange_names],
        "stable_fx_taker_fee": [EXCHANGE_TAKER_FEES_STABLEFX.get(ex, DEFAULT_TAKER_FEE_STABLEFX) for ex in exchange_names],
    })
    df_fees_pretty = prettify_df(df_fees)
    fee_ex_col = [c for c in df_fees_pretty.columns if c.lower() == "exchange"]
    fee_style = df_fees_pretty.style.format({"Standard Taker Fee": "{:.4%}", "Stable Fx Taker Fee": "{:.4%}"}).apply(
        lambda _: style_exchange_cells(df_fees_pretty, fee_ex_col), axis=None
    )
    df_show(fee_style, width="content", hide_index=True)

with top_cols[1]:
    st.subheader("Live FX Feeds")
    if df_fx_summary is None or df_fx_summary.empty:
        st.info("No FX quotes returned from selected exchanges.")
    else:
        df_fx_pretty = prettify_df(df_fx_summary.copy())
        exch_cols = [c for c in df_fx_pretty.columns if c in ("Bid Ex", "Ask Ex")]
        fx_style = df_fx_pretty.style.format({
            "Best Bid": "{:,.6f}", "Best Ask": "{:,.6f}", "Mid": "{:,.6f}", "Spread Bps": "{:,.2f}",
        }).apply(lambda _: style_exchange_cells(df_fx_pretty, exch_cols), axis=None)
        df_show(fx_style, width="stretch", hide_index=True)

with top_cols[2]:
    st.subheader("Current Live Prices")
    st.caption(f"Pair: {pair_for_top}")
    if df_sym_quotes_top is None or df_sym_quotes_top.empty:
        st.info("No quotes returned.")
    else:
        df_mids = df_sym_quotes_top[["exchange", "bid", "ask", "mid"]].copy().sort_values("mid", ascending=False).reset_index(drop=True)
        df_mids_pretty = prettify_df(df_mids)
        ex_cols = [c for c in df_mids_pretty.columns if c.lower() == "exchange"]
        mids_style = df_mids_pretty.style.format({"Bid": "{:,.2f}", "Ask": "{:,.2f}", "Mid": "{:,.2f}"}).apply(
            lambda _: style_exchange_cells(df_mids_pretty, ex_cols), axis=None
        )
        df_show(mids_style, width="content", hide_index=True)

st.divider()

# ============================
# Section 1
# ============================

st.subheader("1) Exchange vs. Exchange Matrix")

pcol1, pcol2 = st.columns([1, 4], gap="large")
with pcol1:
    matrix_symbol = st.selectbox("Select Pair:", options=matrix_universe, index=0, key="matrix_symbol")

with st.spinner(f"Fetching {matrix_symbol} across exchanges..."):
    df_sym_quotes, df_sym_errors = fetch_quotes_for_symbol(tuple(exchange_names), matrix_symbol)

exp = len(exchange_names)
got = 0 if df_sym_quotes.empty else len(df_sym_quotes)
err = 0 if df_sym_errors.empty else len(df_sym_errors)

m1, m2, m3 = st.columns([1, 1, 2])
m1.metric("Expected", f"{exp:,}")
m2.metric("Quotes", f"{got:,}")
m3.metric("Errors / Missing", f"{err:,}")

matrix_vals, present_ex, _mids = build_exchange_matrix_values(df_sym_quotes, exchange_names, matrix_symbol)
disp = matrix_display_table(matrix_vals, present_ex, pair_label=matrix_symbol)

if disp.empty:
    st.info("No exchanges returned this pair.")
else:
    disp_style = disp.style.apply(style_matrix_display, axis=None)
    # no forced height -> no scrollbars
    render_styler_html(disp_style, css_class="matrix_table", hide_thead=True)

st.caption("Matrix cell: Δ = column_mid − row_mid ; bps = Δ / row_mid. Green means column > row; red means column < row.")
st.divider()

# ============================
# Section 2
# ============================

st.subheader("2) Multi Leg Arbitrage Builder (Across Exchanges)")

with st.form("arb_builder_form", border=True):

    cB, cC, cD, cE = st.columns([1.0, 1.0, 1.2, 1.8], gap="medium")
    
    with cB:
        start_ccy = st.selectbox("Start currency", options=["USD", "EUR", "GBP", "JPY", "TRY", "USDT", "USDC", "BTC", "ETH"], index=0)
    with cC:
        end_ccy = st.selectbox("End currency", options=["EUR", "USD", "GBP", "JPY", "TRY", "USDT", "USDC", "BTC", "ETH"], index=0)
    
    with cD:
        order_size_text = st.text_input("Order size (Start units)", value=st.session_state.get("order_size_text", "1,000,000"),
                                        help="You can type commas, e.g. 1,000,000")
        st.session_state["order_size_text"] = order_size_text
        order_size_start = parse_number_with_commas(order_size_text)
        if order_size_start is None:
            st.error("Invalid order size. Example: 1,000,000")
            order_size_start = 0.0
        else:
            st.session_state["order_size_text"] = fmt_int_commas(order_size_start)
    
    with cE:
        st.caption("Best path uses best TOB edge per leg across all selected exchanges (fees optional).")
    
    opt_col1, opt_col2 = st.columns([1.1, 2.9], gap="medium")
    with opt_col1:
        use_standard = st.checkbox("Allow Standard symbols", value=True)
        use_stablefx = st.checkbox("Allow Stable/FX symbols", value=True)
        apply_fees = st.checkbox("Apply taker fees", value=True)
    with opt_col2:
        max_hops = st.slider("Max hops", min_value=1, max_value=6, value=3, step=1)
    
    allowed_symbols: List[str] = []
    if use_standard:
        allowed_symbols += (standard_pick if standard_pick else STANDARD_SYMBOLS)
    if use_stablefx:
        allowed_symbols += (stablefx_pick if stablefx_pick else STABLE_FX_SYMBOLS)
    allowed_symbols = sorted(list(dict.fromkeys(allowed_symbols)))
    
    run_best = st.form_submit_button("Find Best Path", width="content")

if run_best:
    if start_ccy.upper() == end_ccy.upper():
        st.info("Start and end currencies are the same.")
    elif not allowed_symbols:
        st.warning("No symbols allowed. Enable at least one category and/or pick symbols in sidebar.")
    else:
        direct_syms = [f"{start_ccy.upper()}/{end_ccy.upper()}", f"{end_ccy.upper()}/{start_ccy.upper()}"]
        symbols_to_fetch = sorted(list(dict.fromkeys(allowed_symbols + direct_syms)))

        with st.spinner("Fetching quotes for builder..."):
            df_all_quotes, df_all_errors = fetch_quotes_for_symbols_all_exchanges(tuple(exchange_names), tuple(symbols_to_fetch))

        if df_all_quotes.empty:
            st.error("No quotes returned for the selected symbol set across selected exchanges.")
        else:
            edges = build_best_cross_exchange_edges(df_all_quotes, apply_fees=apply_fees)
            if not edges:
                st.error("No usable conversion edges were built (order books empty/missing).")
            else:
                candidates = enumerate_candidate_paths(edges, start_ccy, end_ccy, max_hops=max_hops, per_node_k=12, top_results=50)
                if not candidates:
                    st.warning("No path found. Try increasing max hops or allowing more symbols.")
                else:
                    # ensure best candidate is #1
                    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    best_hops, best_rate, best_path = candidates[0]
                    # Persist results so UI controls (e.g., log-scale toggle) don't wipe results on rerun
                    st.session_state['arb_last_run_key'] = (tuple(exchange_names), tuple(symbols_to_fetch), start_ccy, end_ccy, max_hops, bool(apply_fees))
                    st.session_state['arb_candidates'] = candidates
                    st.session_state['arb_best_path'] = best_path
                    st.session_state['arb_best_rate'] = best_rate
                    st.session_state['arb_start'] = start_ccy
                    st.session_state['arb_end'] = end_ccy

                    st.success(f"Best path rate: 1 {start_ccy} → {best_rate:,.8f} {end_ccy}  (≤ {max_hops} hops)")

                    # ---- Row 1: Leaderboard + Legs ----
                    left, right = st.columns([1.65, 1.35], gap="large")

                    leaderboard_df, leaderboard_paths = build_leaderboard(candidates, start=start_ccy)

                    with left:
                        st.markdown("### Candidate Path Leaderboard")
                        lb_pretty = leaderboard_df.copy()
                        lb_style = lb_pretty.style.format({"Rate (End per 1 Start)": "{:,.10f}"}).apply(style_leaderboard, axis=None)
                        df_show(lb_style, width="stretch", hide_index=True)
                        # Showing legs for the optimal (top-ranked) path
                        selected_path = best_path

                    with right:
                        st.markdown("### Best Path Legs (Optimal)")
                        df_legs = xedges_to_table(selected_path)
                        df_legs_pretty = prettify_df(df_legs)
                        exch_cols = [c for c in df_legs_pretty.columns if c.lower() == "exchange"]
                        legs_style = df_legs_pretty.style.format({
                            "Price Used": "{:,.8f}",
                            "Fee Bps": "{:,.1f}",
                            "To Per 1 From (net)": "{:,.10f}",
                        }).apply(lambda _: style_exchange_cells(df_legs_pretty, exch_cols), axis=None)
                        df_show(legs_style, width="stretch", hide_index=True)

                    st.divider()

                    # ---- Row 2: Size impact vs direct hedge ----
                    st.markdown("### Size Impact vs Direct Hedge")

                    ref_rate, ref_symbol, ref_exchange = best_direct_rate_start_to_end(df_all_quotes, start_ccy, end_ccy, apply_fees=apply_fees)
                    if ref_rate is None:
                        st.info("No direct hedge market found across selected exchanges for this currency pair.")
                    else:
                        best_end_amt = order_size_start * best_rate
                        ref_end_amt = order_size_start * ref_rate
                        profit_end_amt = best_end_amt - ref_end_amt
                        profit_bps = ((best_rate - ref_rate) / ref_rate) / 0.0001 if ref_rate else float("nan")

                        summary = pd.DataFrame([
                            {"scenario": "Best path (optimal)", "rate_end_per_1_start": best_rate, "end_proceeds": best_end_amt},
                            {"scenario": f"Direct hedge ({ref_exchange})", "rate_end_per_1_start": ref_rate, "end_proceeds": ref_end_amt},
                            {"scenario": "Difference", "rate_end_per_1_start": best_rate - ref_rate, "end_proceeds": profit_end_amt},
                        ])
                        summary_pretty = prettify_df(summary)

                        df_show(
                            summary_pretty.style.format({"Rate End Per 1 Start": "{:,.10f}", "End Proceeds": "{:,.2f}"}),
                            width="stretch",
                            hide_index=True,
                        )

                        a, b, c = st.columns(3)
                        a.metric(f"Profit ({end_ccy})", f"{profit_end_amt:,.2f}")
                        b.metric("Profit (bps vs hedge)", f"{profit_bps:,.2f}")
                        c.metric("Direct hedge symbol", f"{ref_symbol}")

                    st.divider()

                    # ---- Row 3: Big full-width path map ----
                    
# ---- Row 3: Big full-width rate progression chart ----
if st.session_state.get("arb_candidates") and st.session_state.get("arb_best_path"):
    candidates = st.session_state["arb_candidates"]
    best_path = st.session_state["arb_best_path"]
    best_rate = st.session_state.get("arb_best_rate")
    start_ccy = st.session_state.get("arb_start", start_ccy)
    end_ccy = st.session_state.get("arb_end", end_ccy)
    st.markdown("### Arbitrage Rate Progression (Top Candidates)")
    st.caption("Each line shows how 1 unit of the Start currency compounds across legs. The best path will end highest.")

    use_log = st.checkbox("Log scale (better for tiny edges)", value=False, key="rate_progress_log")

    pts = build_rate_progress_points_from_candidates(start_ccy, candidates, best_path=best_path)

    # Clean labeling: label each path once, at its final point
    if pts is not None and not pts.empty:
        pts = pts.copy()
        pts["path_label"] = ""
        mask_final = pts["is_final"] == True

        def _mk_label(r):
            fr = None
            try:
                fr = float(r.get("final_rate"))
            except Exception:
                fr = None
            if bool(r.get("is_best", False)):
                return f"BEST  {fr:,.16f}" if fr is not None else "BEST"
            pid = str(r.get("path_id"))
            return f"{pid}  {fr:,.16f}" if fr is not None else pid

        if mask_final.any():
            pts.loc[mask_final, "path_label"] = pts.loc[mask_final].apply(_mk_label, axis=1)

    y_field = "log_cum_rate" if use_log else "cum_rate"
    y_title = "log(Cumulative rate)" if use_log else f"Cumulative {end_ccy} per 1 {start_ccy}"

    # Tight y-domain so small differences are visually distinguishable.
    y_min = float(pts[y_field].min()) if pts is not None and not pts.empty else None
    y_max = float(pts[y_field].max()) if pts is not None and not pts.empty else None
    y_pad = None
    if y_min is not None and y_max is not None:
        span = y_max - y_min
        if span <= 0:
            y_pad = max(abs(y_max) * 1e-6, 1e-12)
        else:
            y_pad = span * 0.06
    y_scale = {"zero": False, "nice": False}
    if y_min is not None and y_max is not None and y_pad is not None:
        y_scale["domain"] = [y_min - y_pad, y_max + y_pad]

    # --- Axis ticks: optionally force y-axis ticks to be EXACTLY the final path rates (as shown in the leaderboard).
    # This makes each path end at a visually distinct, labeled tick value.
    y_axis = {
        "labelColor": "#e5e7eb",
        "titleColor": "#e5e7eb",
        "labelOverlap": "greedy",
        # Keep axis readable; exact final rates are labeled on the last point of each path.
        "format": ",.8f",
    }

    # Vega-Lite: one line per path, labels only on final points
    spec = {
        "width": "container",
        "height": 560,
        "config": {
            "background": "#0b1020",
            "axis": {
                "labelColor": "#e5e7eb",
                "titleColor": "#e5e7eb",
                "gridColor": "#1f2937",
                "domainColor": "#334155",
                "tickColor": "#334155",
            },
            "view": {"stroke": None},
        },
        "layer": [
            {
                "mark": {"type": "line", "point": False},
                "encoding": {
                    "x": {
                        "field": "step",
                        "type": "quantitative",
                        "title": "Step",
                        "axis": {"tickMinStep": 1},
                    },
                    "y": {
                        "field": y_field,
                        "type": "quantitative",
                        "title": y_title,
                        "scale": y_scale,
                        "axis": y_axis if not use_log else {},
                    },
                    "detail": {"field": "path_id"},
                    "order": {"field": "step"},
                    "opacity": {
                        "condition": {"test": "datum.is_best == true", "value": 1.0},
                        "value": 0.20
                    },
                    "strokeWidth": {
                        "condition": {"test": "datum.is_best == true", "value": 4},
                        "value": 1.5
                    },
                    "tooltip": [
                        {"field": "path_id", "type": "nominal", "title": "Path"},
                        {"field": "step", "type": "quantitative", "title": "Step"},
                        {"field": "cum_rate", "type": "quantitative", "format": ",.10f", "title": "Cum rate"},
                        {"field": "final_rate", "type": "quantitative", "format": ",.10f", "title": "Final rate"},
                        {"field": "asset_path", "type": "nominal", "title": "Asset Path"},
                        {"field": "venue_path", "type": "nominal", "title": "Venue Path"},
                    ],
                },
            },
            {
                "transform": [{"filter": "datum.is_final == true"}],
                "mark": {"type": "text", "align": "left", "dx": 8, "dy": -8, "fontSize": 12, "color": "#e5e7eb"},
                "encoding": {
                    "x": {"field": "step", "type": "quantitative"},
                    "y": {"field": y_field, "type": "quantitative", "scale": y_scale},
                    "text": {"field": "path_label", "type": "nominal"},
                    "opacity": {
                        "condition": {"test": "datum.is_best == true", "value": 1.0},
                        "value": 0.55
                    },
                },
            }
        ],
    }

    if pts is not None and not pts.empty:
        st.vega_lite_chart(pts, spec, width="stretch")
    else:
        st.info("No candidate paths to chart.")
    
else:
    st.info("Run **Find Best Path** to populate the arbitrage chart.")
st.caption("TOB only; no depth. Fees are assumptions. Executable arb depends on size, latency, and venue constraints.")