import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime
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

    # Divergence in percentage points:
    # Negative divergence => heavyweights falling more than the index.
    divergence = avg_heavy_change - nifty_change

    # Very rough "pressure" in points (just for intuition):
    # If heavyweights were moving with Nifty, index move would be closer to avg_heavy_change.
    # Here we only show divergence in % terms & classify.
    suppression_label = "NORMAL"
    suppression_explanation = "No clear sign of heavy index suppression."

    # Heuristic: we look for situations where Nifty is modestly down but
    # heavyweights are noticeably weaker.
    abs_div = abs(divergence)

    if (
        nifty_change <= -0.20   # Nifty at least mildly down
        and nifty_change >= -1.50  # not a full crash, more like intraday dip
        and divergence <= -0.30  # heavyweights underperforming Nifty by > 0.30%
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


# --------- LAYOUT HELPERS ---------
def layout_header():
    st.title("NIFTY Operator Detector – Phase 2")
    st.caption(
        "Live snapshot of NIFTY 50 and top heavyweights, plus a basic "
        "operator-suppression monitor. Next phases will add ATM option & order-book checks."
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


def layout_snapshot(df: pd.DataFrame):
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

    # Suppression section right under Nifty snapshot
    layout_suppression_section(df)

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
            "Phase 3 will add ATM option checks; Phase 4 will add order-book signals."
        )

    kite = get_kite_client()

    def run_fetch_and_render():
        try:
            df = fetch_ltp_snapshot(kite)
        except Exception as e:
            st.error(f"Error fetching LTP snapshot from Kite: {e}")
            return
        layout_snapshot(df)

    run_fetch_and_render()

    st.info(
        "Hit the **R** key (browser refresh) or click the rerun button in the top-right "
        "to refresh the snapshot."
    )


if __name__ == "__main__":
    main()
