import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo


# --------- CONFIG: HEAVYWEIGHTS & INDEX ---------
HEAVYWEIGHT_SYMBOLS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "ITC",
    "LT",
    "SBIN",
    "BHARTIARTL",
    "HINDUNILVR",
]

NIFTY_INDEX_SYMBOL = "NIFTY 50"  # Kite uses "NSE:NIFTY 50" as instrument


# --------- KITE CLIENT HELPERS ---------
@st.cache_resource(show_spinner=False)
def get_kite_client():
    """
    Create and cache a KiteConnect client using API key + access token
    from Streamlit secrets.
    """
    try:
        api_key = st.secrets["kite"]["api_key"]
        access_token = st.secrets["kite"]["access_token"]
    except Exception:
        st.error(
            "Kite API credentials not found in Streamlit secrets. "
            "Go to app settings → Secrets and add:\n\n"
            "[kite]\napi_key = \"...\"\naccess_token = \"...\""
        )
        st.stop()

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def get_nifty_option_instruments(kite: KiteConnect) -> pd.DataFrame:
    """
    Fetch NIFTY option instruments from NFO.
    Filter only OPTIDX NIFTY contracts and pre-process expiry.
    Not cached because KiteConnect object is not hashable.
    """
    try:
        instruments = kite.instruments("NFO")
    except Exception as e:
        st.error(f"Error fetching NFO instruments from Kite: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(instruments)

    if df.empty:
        return df

    # Filter for NIFTY index options only
    df = df[
        (df["name"] == "NIFTY")
        & (df["segment"] == "NFO-OPT")
        & (df["instrument_type"] == "OPTIDX")
    ].copy()

    if df.empty:
        return df

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


def build_instrument_list():
    """
    Build the list of instruments to query via ltp().
    """
    instruments = [f"NSE:{sym}" for sym in HEAVYWEIGHT_SYMBOLS]
    instruments.append(f"NSE:{NIFTY_INDEX_SYMBOL}")
    return instruments


def fetch_ltp_snapshot(kite: KiteConnect) -> pd.DataFrame:
    """
    Fetch LTP + previous close for NIFTY and heavyweights.
    Returns a pandas DataFrame.
    """
    instruments = build_instrument_list()
    ltp_data = kite.ltp(instruments)

    rows = []
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    for instrument, data in ltp_data.items():
        # instrument is like "NSE:RELIANCE"
        _, symbol = instrument.split(":", 1)

        last_price = data.get("last_price")
        ohlc = data.get("ohlc", {})
        prev_close = ohlc.get("close")

        # Safely compute % change
        if last_price is not None and prev_close not in (None, 0):
            pct_change = ((last_price - prev_close) / prev_close) * 100.0
        else:
            pct_change = None

        rows.append(
            {
                "Symbol": symbol,
                "LTP": last_price,
                "Prev Close": prev_close,
                "% Change": pct_change,
                "Timestamp": now,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Flag NIFTY row and sort it on top, then biggest movers
    df["is_nifty"] = df["Symbol"].eq(NIFTY_INDEX_SYMBOL)
    df = df.sort_values(
        by=["is_nifty", "% Change"],
        ascending=[False, False],
        ignore_index=True,
    ).drop(columns=["is_nifty"])

    return df


# --------- SUPPRESSION LOGIC ---------
def compute_suppression_stats(df: pd.DataFrame):
    """
    Compute basic operator-suppression style metrics:
    - Nifty % change
    - Average heavyweights % change
    - Divergence (heavyweights - Nifty)
    - A simple suppression label
    """
    if df is None or df.empty:
        return None

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return None

    nifty_row = nifty_rows.iloc[0]
    heavy_df = df[df["Symbol"] != NIFTY_INDEX_SYMBOL]

    if heavy_df.empty:
        return None

    nifty_change = nifty_row["% Change"]
    avg_heavy_change = heavy_df["% Change"].mean()

    if pd.isna(nifty_change) or pd.isna(avg_heavy_change):
        return None

    divergence = avg_heavy_change - nifty_change

    suppression_label = "NORMAL"
    suppression_explanation = "No clear sign of heavy index suppression."

    abs_div = abs(divergence)

    if (
        nifty_change <= -0.20   # Nifty at least mildly down
        and nifty_change >= -1.50  # not a crash
        and divergence <= -0.30  # heavyweights weaker than Nifty
    ):
        if abs_div >= 1.0:
            suppression_label = "HIGH"
            suppression_explanation = (
                "Heavyweights are significantly weaker than NIFTY – "
                "possible deliberate suppression while the index looks 'orderly'."
            )
        else:
            suppression_label = "MILD"
            suppression_explanation = (
                "Heavyweights are weaker than NIFTY – some index suppression pressure visible."
            )

    return {
        "nifty_change": float(nifty_change),
        "avg_heavy_change": float(avg_heavy_change),
        "divergence": float(divergence),
        "label": suppression_label,
        "explanation": suppression_explanation,
    }


# --------- ATM CE FINDER & DIVERGENCE ---------
def find_atm_ce_instrument(nifty_opt_df: pd.DataFrame, nifty_spot: float):
    """
    Given pre-filtered NIFTY options df and current spot,
    find nearest ATM CE for the nearest expiry >= today.
    """
    if nifty_opt_df is None or nifty_opt_df.empty:
        return None

    if nifty_spot is None or pd.isna(nifty_spot):
        return None

    # NIFTY strikes are in 50-point increments
    atm_strike = round(nifty_spot / 50.0) * 50

    today = date.today()
    ce_df = nifty_opt_df[
        (nifty_opt_df["strike"] == atm_strike)
        & (nifty_opt_df["instrument_type"] == "OPTIDX")
    ].copy()

    if ce_df.empty:
        return None

    ce_df = ce_df[ce_df["expiry"] >= today]
    if ce_df.empty:
        return None

    ce_df = ce_df.sort_values("expiry")
    return ce_df.iloc[0]


def fetch_atm_ce_quote(kite: KiteConnect, nifty_opt_df: pd.DataFrame, nifty_spot: float):
    """
    Find ATM CE instrument and fetch its LTP + % change.
    Returns a dict with option details or None.
    """
    row = find_atm_ce_instrument(nifty_opt_df, nifty_spot)
    if row is None:
        return None

    tradingsymbol = row["tradingsymbol"]
    strike = row["strike"]
    expiry = row["expiry"]
    instrument = f"NFO:{tradingsymbol}"

    try:
        q = kite.ltp([instrument])
    except Exception as e:
        st.error(f"Error fetching ATM CE LTP from Kite: {e}")
        return None

    if not q:
        return None

    data = list(q.values())[0]
    last_price = data.get("last_price")
    ohlc = data.get("ohlc", {})
    prev_close = ohlc.get("close")

    if last_price is not None and prev_close not in (None, 0):
        pct_change = ((last_price - prev_close) / prev_close) * 100.0
    else:
        pct_change = None

    return {
        "tradingsymbol": tradingsymbol,
        "strike": float(strike),
        "expiry": expiry,
        "ltp": last_price,
        "prev_close": prev_close,
        "pct_change": pct_change,
        "instrument": instrument,
    }


# --------- ORDER BOOK / DEPTH LOGIC ---------
def fetch_orderbook_for_atm_ce(kite: KiteConnect, atm_ce_info: dict):
    """
    Fetch depth (order book) for the ATM CE and compute:
    - total bid qty
    - total ask qty
    - top bid/ask prices
    - bid dominance ratio
    """
    if atm_ce_info is None:
        return None

    instrument = atm_ce_info.get("instrument")
    if not instrument:
        return None

    try:
        # quote() gives depth with buy/sell ladders
        q = kite.quote([instrument])
    except Exception as e:
        st.error(f"Error fetching order book for ATM CE from Kite: {e}")
        return None

    if not q:
        return None

    data = list(q.values())[0]
    depth = data.get("depth", {})
    buy_levels = depth.get("buy", []) or []
    sell_levels = depth.get("sell", []) or []

    total_bid_qty = sum(level.get("quantity", 0) for level in buy_levels)
    total_ask_qty = sum(level.get("quantity", 0) for level in sell_levels)

    top_bid_price = buy_levels[0]["price"] if buy_levels else None
    top_ask_price = sell_levels[0]["price"] if sell_levels else None

    if total_ask_qty <= 0:
        bid_dom_ratio = float("inf") if total_bid_qty > 0 else 0.0
    else:
        bid_dom_ratio = total_bid_qty / total_ask_qty

    # Classification heuristic:
    # - STRONG: bid_dom_ratio >= 2
    # - MILD:   1.2 <= bid_dom_ratio < 2
    # - NONE:   otherwise
    if bid_dom_ratio >= 2.0 and total_bid_qty > 0:
        footprint = "STRONG"
        desc = "Bid side is heavily stacked vs ask – strong buying footprint."
    elif bid_dom_ratio >= 1.2 and total_bid_qty > 0:
        footprint = "MILD"
        desc = "Bid side is somewhat dominant – mild buying footprint."
    else:
        footprint = "NONE"
        desc = "No special dominance on bid side."

    return {
        "total_bid_qty": total_bid_qty,
        "total_ask_qty": total_ask_qty,
        "top_bid_price": top_bid_price,
        "top_ask_price": top_ask_price,
        "bid_dom_ratio": bid_dom_ratio,
        "footprint": footprint,
        "description": desc,
    }


# --------- LAYOUT HELPERS ---------
def layout_header():
    st.title("NIFTY Operator Detector – Phase 4")
    st.caption(
        "Heavyweight suppression + ATM Call divergence + order-book footprint.\n"
        "Use this to spot suppressed indices with aggressive call accumulation."
    )


def layout_suppression_section(df: pd.DataFrame):
    stats = compute_suppression_stats(df)
    st.subheader("🧲 Operator Suppression Monitor")

    if stats is None:
        st.info("Not enough clean data yet to compute suppression metrics.")
        return

    nifty_chg = stats["nifty_change"]
    heavy_chg = stats["avg_heavy_change"]
    divergence = stats["divergence"]
    label = stats["label"]
    explanation = stats["explanation"]

    # Safe formatting
    def fmt_pct(x):
        try:
            return f"{x:.2f}%"
        except Exception:
            return "-"

    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY % Change", fmt_pct(nifty_chg))
    col2.metric("Avg Heavyweights % Change", fmt_pct(heavy_chg))
    col3.metric("Divergence (Heavy - NIFTY)", fmt_pct(divergence))

    if label == "HIGH":
        st.error(f"Suppression: {label} – {explanation}")
    elif label == "MILD":
        st.warning(f"Suppression: {label} – {explanation}")
    else:
        st.success(f"Suppression: {label} – {explanation}")


def layout_atm_ce_section(atm_ce_info, ob_info, nifty_change):
    st.subheader("🎯 ATM Call Divergence & Order Book")

    if atm_ce_info is None:
        st.info("ATM NIFTY CE data not available yet.")
        return

    if nifty_change is None or pd.isna(nifty_change):
        st.info("NIFTY % change unavailable; cannot compute divergence.")
        return

    ce_chg = atm_ce_info["pct_change"]
    ce_ltp = atm_ce_info["ltp"]
    ce_symbol = atm_ce_info["tradingsymbol"]
    strike = atm_ce_info["strike"]
    expiry = atm_ce_info["expiry"]

    # Safe formatting helpers
    def fmt_price(x):
        if x is None or pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "-"

    def fmt_pct(x):
        if x is None or pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "-"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATM CE", f"{ce_symbol}")
    col2.metric("Strike", f"{int(strike)}")
    col3.metric("Expiry", str(expiry))
    col4.metric("CE LTP", fmt_price(ce_ltp))

    col5, col6 = st.columns(2)
    col5.metric("CE % Change", fmt_pct(ce_chg))
    col6.metric("NIFTY % Change", fmt_pct(nifty_change))

    # Classify divergence: NIFTY weak, CE resilient/strong
    note = "No special divergence detected."
    div_style = "neutral"

    if ce_chg is not None and not pd.isna(ce_chg):
        ce_chg_f = float(ce_chg)
        nifty_f = float(nifty_change)

        # Scenarios:
        # 1) Nifty down, CE flat/up => strong call accumulation footprint
        # 2) Nifty down, CE down much less than underlying => mild accumulation
        if nifty_f <= -0.20:
            if ce_chg_f >= 0.0:
                div_style = "strong"
                note = (
                    "NIFTY is weak, but ATM CE is flat or rising – "
                    "possible aggressive call accumulation."
                )
            elif ce_chg_f > nifty_f + 3.0:
                # e.g., Nifty -0.8%, CE only -0.2% => CE holding up
                div_style = "mild"
                note = (
                    "NIFTY is weak, but ATM CE is holding up better – "
                    "mild sign of supportive call buying."
                )

    # Show divergence signal
    if div_style == "strong":
        st.error(f"Divergence Signal: STRONG – {note}")
    elif div_style == "mild":
        st.warning(f"Divergence Signal: MILD – {note}")
    else:
        st.info(f"Divergence Signal: NEUTRAL – {note}")

    # ---- ORDER BOOK FOOTPRINT ----
    st.markdown("---")
    st.markdown("**Order Book Footprint (ATM CE)**")

    if ob_info is None:
        st.info("Order book data not available.")
        return

    total_bid = ob_info["total_bid_qty"]
    total_ask = ob_info["total_ask_qty"]
    top_bid = ob_info["top_bid_price"]
    top_ask = ob_info["top_ask_price"]
    ratio = ob_info["bid_dom_ratio"]
    footprint = ob_info["footprint"]
    desc = ob_info["description"]

    colb1, colb2, colb3, colb4 = st.columns(4)
    colb1.metric("Total Bid Qty (top 5)", f"{int(total_bid)}")
    colb2.metric("Total Ask Qty (top 5)", f"{int(total_ask)}")
    colb3.metric(
        "Bid/Aks Qty Ratio",
        "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}",
    )
    colb4.metric("Top Bid / Ask", f"{fmt_price(top_bid)} / {fmt_price(top_ask)}")

    if footprint == "STRONG":
        st.error(f"Big Buyer Footprint: STRONG – {desc}")
    elif footprint == "MILD":
        st.warning(f"Big Buyer Footprint: MILD – {desc}")
    else:
        st.info(f"Big Buyer Footprint: NONE – {desc}")


def layout_snapshot(df: pd.DataFrame, atm_ce_info, ob_info):
    if df is None or df.empty:
        st.warning("No data returned from Kite API.")
        return

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]

    if nifty_rows.empty:
        st.warning(
            "NIFTY 50 row not found in data. Showing only heavyweights table."
        )
        _render_heavyweights_table(df)
        return

    nifty_row = nifty_rows.iloc[0]

    nifty_ltp = nifty_row["LTP"]
    nifty_change = nifty_row["% Change"]
    nifty_ts = nifty_row["Timestamp"]

    # Safe formatting for metrics
    if pd.isna(nifty_ltp):
        nifty_ltp_display = "-"
    else:
        nifty_ltp_display = f"{float(nifty_ltp):.2f}"

    if pd.isna(nifty_change):
        nifty_change_display = "-"
    else:
        nifty_change_display = f"{float(nifty_change):.2f}%"

    if isinstance(nifty_ts, datetime):
        ts_display = nifty_ts.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
    else:
        ts_display = str(nifty_ts)

    st.subheader("📈 NIFTY Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY LTP", nifty_ltp_display)
    col2.metric("NIFTY % Change", nifty_change_display)
    col3.write(f"Timestamp (IST): {ts_display}")

    # Suppression section
    layout_suppression_section(df)

    # ATM CE divergence + order book section
    layout_atm_ce_section(atm_ce_info, ob_info, nifty_change)

    st.subheader("🏋️ Heavyweights Snapshot")
    _render_heavyweights_table(df)


def _render_heavyweights_table(df: pd.DataFrame):
    display_df = df.copy()

    # Format columns safely
    def fmt_price(x):
        if pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "-"

    def fmt_pct(x):
        if pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "-"

    display_df["LTP"] = display_df["LTP"].map(fmt_price)
    display_df["Prev Close"] = display_df["Prev Close"].map(fmt_price)
    display_df["% Change"] = display_df["% Change"].map(fmt_pct)

    st.dataframe(
        display_df[["Symbol", "LTP", "Prev Close", "% Change"]],
        use_container_width=True,
        hide_index=True,
    )


# --------- MAIN APP ---------
def main():
    layout_header()

    with st.sidebar:
        st.header("Settings")
        _ = st.slider("Auto-refresh every (seconds)", 5, 60, 15)
        st.caption(
            "For now, refresh manually with the rerun button or browser reload.\n"
            "Use this as a discretionary tool, not a blind auto-trader."
        )

    kite = get_kite_client()
    nifty_opt_df = get_nifty_option_instruments(kite)

    def run_fetch_and_render():
        try:
            df = fetch_ltp_snapshot(kite)
        except Exception as e:
            st.error(f"Error fetching LTP snapshot from Kite: {e}")
            return

        # Get NIFTY spot for ATM logic
        nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
        atm_ce_info = None
        ob_info = None

        if not nifty_rows.empty:
            nifty_spot = nifty_rows.iloc[0]["LTP"]
            if (
                nifty_spot is not None
                and not pd.isna(nifty_spot)
                and not nifty_opt_df.empty
            ):
                atm_ce_info_local = fetch_atm_ce_quote(
                    kite, nifty_opt_df, float(nifty_spot)
                )
                atm_ce_info = atm_ce_info_local

                if atm_ce_info is not None:
                    ob_info_local = fetch_orderbook_for_atm_ce(kite, atm_ce_info)
                    ob_info = ob_info_local

        layout_snapshot(df, atm_ce_info, ob_info)

    run_fetch_and_render()

    st.info(
        "Hit the **R** key (browser refresh) or click the rerun button in the top-right "
        "to refresh the snapshot."
    )


if __name__ == "__main__":
    main()
