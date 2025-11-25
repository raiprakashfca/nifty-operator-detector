import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime


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
    except Exception as e:
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
    # Example: ["NSE:RELIANCE", ..., "NSE:NIFTY 50"]
    ltp_data = kite.ltp(instruments)

    rows = []
    now = datetime.now()

    for instrument, data in ltp_data.items():
        # instrument is like "NSE:RELIANCE"
        _, symbol = instrument.split(":", 1)

        last_price = data.get("last_price")
        ohlc = data.get("ohlc", {})
        prev_close = ohlc.get("close")

        # Safely compute % change
        if (
            last_price is not None
            and prev_close not in (None, 0)
        ):
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


def layout_header():
    st.title("NIFTY Operator Detector – Phase 1")
    st.caption(
        "Live snapshot of NIFTY 50 and top heavyweights – this is the data spine "
        "for spotting fake suppression and cheap-call zones."
    )


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
        nifty_ltp_display = f"{nifty_ltp:.2f}"

    if pd.isna(nifty_change):
        nifty_change_display = "-"
    else:
        nifty_change_display = f"{nifty_change:.2f}%"

    ts_display = nifty_ts.strftime("%H:%M:%S") if isinstance(nifty_ts, datetime) else str(nifty_ts)

    st.subheader("📈 NIFTY Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY LTP", nifty_ltp_display)
    col2.metric("NIFTY % Change", nifty_change_display)
    col3.write(f"Timestamp: {ts_display}")

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


def main():
    layout_header()

    with st.sidebar:
        st.header("Settings")
        refresh_seconds = st.slider("Auto-refresh every (seconds)", 5, 60, 15)
        st.caption(
            "For Phase 1 we only show LTP & % change. "
            "In the next phases we'll add operator-pressure signals here."
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
        "to refresh based on your selected interval."
    )
    st.caption(
        "Later we'll replace this with a smarter auto-refresh and signal engine."
    )


if __name__ == "__main__":
    main()
