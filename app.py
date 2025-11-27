import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="NIFTY Operator Detector",
    layout="wide",
)

# --------- BURST MODE CONFIG ---------
BURST_REFRESH_SECONDS = 2  # when strong operator footprint is detected


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
def get_kite_client() -> KiteConnect:
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


@st.cache_data(show_spinner=False, ttl=3600)
def get_nifty_option_instruments() -> pd.DataFrame:
    """
    Fetch and cache NIFTY option instruments from NFO for 1 hour.

    In Kite's instruments dump:
      - exchange: 'NFO'
      - segment: 'NFO-OPT'
      - name: 'NIFTY'
      - instrument_type: 'CE' or 'PE' for options
    """
    kite = get_kite_client()
    try:
        instruments = kite.instruments("NFO")
    except Exception as e:
        st.error(f"Error fetching NFO instruments from Kite: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(instruments)
    if df.empty:
        st.warning("NFO instruments list is empty.")
        return df

    # Filter for NIFTY index options (CE + PE)
    df = df[
        (df["exchange"] == "NFO")
        & (df["segment"] == "NFO-OPT")
        & (df["name"] == "NIFTY")
    ].copy()

    if df.empty:
        st.warning("No NIFTY option instruments found after filtering.")
        return df

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["strike"] = df["strike"].astype(float)

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

    try:
        ltp_data = kite.ltp(instruments)
    except Exception as e:
        st.error(f"Error fetching LTP data from Kite: {e}")
        return pd.DataFrame()

    rows = []
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    for instrument, data in ltp_data.items():
        # instrument is like "NSE:RELIANCE"
        try:
            _, symbol = instrument.split(":", 1)
        except ValueError:
            symbol = instrument

        last_price = data.get("last_price")
        ohlc = data.get("ohlc", {}) or {}
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


# --------- SUPPRESSION / INFLATION LOGIC ---------
def compute_suppression_stats(df: pd.DataFrame):
    """
    Compute operator-style metrics:
    - Nifty % change
    - Avg heavyweights % change
    - Divergence (heavyweights - Nifty)
    - Suppression label (for down moves)
    - Inflation label (for up moves)
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
    inflation_label = "NORMAL"
    inflation_explanation = "No clear sign of index over-inflation."

    abs_div = abs(divergence)

    # Suppression: NIFTY mildly down, heavyweights materially weaker
    if (
        nifty_change <= -0.20   # mild down
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

    # Inflation: NIFTY mildly up, heavyweights materially stronger
    if (
        nifty_change >= 0.20    # mild up
        and nifty_change <= 1.50  # not a euphoric breakout
        and divergence >= 0.30   # heavyweights stronger than Nifty
    ):
        if abs_div >= 1.0:
            inflation_label = "HIGH"
            inflation_explanation = (
                "Heavyweights are doing much more lifting than NIFTY – "
                "possible deliberate index inflation."
            )
        else:
            inflation_label = "MILD"
            inflation_explanation = (
                "Heavyweights are stronger than NIFTY – some over-inflation pressure visible."
            )

    return {
        "nifty_change": float(nifty_change),
        "avg_heavy_change": float(avg_heavy_change),
        "divergence": float(divergence),
        "supp_label": suppression_label,
        "supp_expl": suppression_explanation,
        "infl_label": inflation_label,
        "infl_expl": inflation_explanation,
    }


# --------- ATM OPTION FINDERS & QUOTES ---------
def find_atm_option_instrument(
    nifty_opt_df: pd.DataFrame, nifty_spot: float, option_type: str
):
    """
    Generic ATM finder for CE/PE.
    option_type: "CE" or "PE"
    """
    if nifty_opt_df is None or nifty_opt_df.empty:
        return None

    if nifty_spot is None or pd.isna(nifty_spot):
        return None

    option_type = option_type.upper()
    if option_type not in ("CE", "PE"):
        return None

    # NIFTY strikes are in 50-point increments
    atm_strike = round(nifty_spot / 50.0) * 50
    today = date.today()

    df = nifty_opt_df[
        (nifty_opt_df["instrument_type"] == option_type)
        & (nifty_opt_df["expiry"] >= today)
    ].copy()

    if df.empty:
        return None

    # Prefer exact ATM strike, else nearest strike
    df["strike_diff"] = (df["strike"] - atm_strike).abs()
    df = df.sort_values(["strike_diff", "expiry"])
    return df.iloc[0]


def fetch_atm_option_quote(
    kite: KiteConnect, nifty_opt_df: pd.DataFrame, nifty_spot: float, option_type: str
):
    """
    Find ATM option (CE or PE) and fetch its LTP + % change.
    Returns a dict with option details or None.
    """
    row = find_atm_option_instrument(nifty_opt_df, nifty_spot, option_type)
    if row is None:
        return None

    tradingsymbol = row["tradingsymbol"]
    strike = row["strike"]
    expiry = row["expiry"]
    instrument = f"NFO:{tradingsymbol}"

    try:
        q = kite.ltp([instrument])
    except Exception as e:
        st.error(f"Error fetching ATM {option_type} LTP from Kite: {e}")
        return None

    if not q:
        return None

    data = list(q.values())[0]
    last_price = data.get("last_price")
    ohlc = data.get("ohlc", {}) or {}
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
        "option_type": option_type.upper(),
    }


# --------- ORDER BOOK / DEPTH LOGIC ---------
def fetch_orderbook_for_option(kite: KiteConnect, opt_info: dict):
    """
    Fetch depth (order book) for an option and compute:
    - total bid qty
    - total ask qty
    - top bid/ask prices
    - bid dominance ratio
    """
    if opt_info is None:
        return None

    instrument = opt_info.get("instrument")
    if not instrument:
        return None

    try:
        q = kite.quote([instrument])
    except Exception as e:
        st.error(f"Error fetching order book from Kite: {e}")
        return None

    if not q:
        return None

    data = list(q.values())[0]
    depth = data.get("depth", {}) or {}
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


# --------- SMALL FORMAT HELPERS ---------
def _fmt_price(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "-"


# --------- DIVERGENCE CLASSIFIERS (used for UI + Burst Mode) ---------
def classify_ce_divergence(ce_chg, nifty_change) -> str:
    """
    Return 'strong', 'mild', or 'neutral' for CE divergence.
    CE logic: NIFTY weak, CE resilient/green.
    """
    if ce_chg is None or pd.isna(ce_chg):
        return "neutral"
    if nifty_change is None or pd.isna(nifty_change):
        return "neutral"

    ce = float(ce_chg)
    nf = float(nifty_change)

    if nf <= -0.20:  # Nifty weak
        if ce >= 0.0:
            return "strong"
        elif ce > nf + 3.0:
            return "mild"

    return "neutral"


def classify_pe_divergence(pe_chg, nifty_change) -> str:
    """
    Return 'strong', 'mild', or 'neutral' for PE divergence.
    PE logic: NIFTY strong, PE resilient/green.
    """
    if pe_chg is None or pd.isna(pe_chg):
        return "neutral"
    if nifty_change is None or pd.isna(nifty_change):
        return "neutral"

    pe = float(pe_chg)
    nf = float(nifty_change)

    if nf >= 0.20:  # Nifty strong
        if pe >= 0.0:
            # Nifty up, PE flat/green -> very odd
            return "strong"
        elif pe > -3.0:
            # Nifty up, PE only mildly red -> not collapsing
            return "mild"

    return "neutral"


# --------- LAYOUT HELPERS ---------
def layout_header():
    st.title("NIFTY Operator Detector – Calls & Puts (Burst Mode)")
    st.caption(
        "Detect two games:\n"
        "1) Index suppression + Call accumulation (buy CE on dips)\n"
        "2) Index inflation + Put accumulation (buy PE on ramps)\n\n"
        "Burst Mode: refresh auto-speeds to 2s when a strong operator footprint appears."
    )


def layout_suppression_section(df: pd.DataFrame):
    stats = compute_suppression_stats(df)
    st.subheader("🧲 Heavyweight vs Index Monitor")

    if stats is None:
        st.info("Not enough clean data yet to compute suppression / inflation metrics.")
        return

    nifty_chg = stats["nifty_change"]
    heavy_chg = stats["avg_heavy_change"]
    divergence = stats["divergence"]
    supp_label = stats["supp_label"]
    supp_expl = stats["supp_expl"]
    infl_label = stats["infl_label"]
    infl_expl = stats["infl_expl"]

    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY % Change", _fmt_pct(nifty_chg))
    col2.metric("Avg Heavyweights % Change", _fmt_pct(heavy_chg))
    col3.metric("Divergence (Heavy - NIFTY)", _fmt_pct(divergence))

    # Suppression message
    if supp_label == "HIGH":
        st.error(f"Suppression: {supp_label} – {supp_expl}")
    elif supp_label == "MILD":
        st.warning(f"Suppression: {supp_label} – {supp_expl}")
    else:
        st.info(f"Suppression: {supp_label} – {supp_expl}")

    # Inflation message
    if infl_label == "HIGH":
        st.error(f"Inflation: {infl_label} – {infl_expl}")
    elif infl_label == "MILD":
        st.warning(f"Inflation: {infl_label} – {infl_expl}")
    else:
        st.info(f"Inflation: {infl_label} – {infl_expl}")


def layout_atm_ce_section(atm_ce_info, ob_info, nifty_change):
    st.subheader("🎯 ATM Call (CE) – Dip-Buying Detector")

    if atm_ce_info is None:
        st.info("ATM NIFTY CE data not available yet.")
        return

    if nifty_change is None or pd.isna(nifty_change):
        st.info("NIFTY % change unavailable; cannot compute CE divergence.")
        return

    ce_chg = atm_ce_info["pct_change"]
    ce_ltp = atm_ce_info["ltp"]
    ce_symbol = atm_ce_info["tradingsymbol"]
    strike = atm_ce_info["strike"]
    expiry = atm_ce_info["expiry"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATM CE", f"{ce_symbol}")
    col2.metric("Strike", f"{int(strike)}")
    col3.metric("Expiry", str(expiry))
    col4.metric("CE LTP", _fmt_price(ce_ltp))

    col5, col6 = st.columns(2)
    col5.metric("CE % Change", _fmt_pct(ce_chg))
    col6.metric("NIFTY % Change", _fmt_pct(nifty_change))

    level = classify_ce_divergence(ce_chg, nifty_change)

    if level == "strong":
        note = (
            "NIFTY is weak, but ATM CE is flat or rising – "
            "possible aggressive call accumulation on dips."
        )
        st.error(f"Call Divergence: STRONG – {note}")
    elif level == "mild":
        note = (
            "NIFTY is weak, but ATM CE is holding up better – "
            "mild supportive call buying."
        )
        st.warning(f"Call Divergence: MILD – {note}")
    else:
        note = "No special CE divergence detected."
        st.info(f"Call Divergence: NEUTRAL – {note}")

    st.markdown("**Order Book Footprint (ATM CE)**")

    if ob_info is None:
        st.info("Order book data for CE not available.")
        return

    total_bid = ob_info["total_bid_qty"]
    total_ask = ob_info["total_ask_qty"]
    ratio = ob_info["bid_dom_ratio"]
    footprint = ob_info["footprint"]
    desc = ob_info["description"]
    top_bid = ob_info["top_bid_price"]
    top_ask = ob_info["top_ask_price"]

    colb1, colb2, colb3, colb4 = st.columns(4)
    colb1.metric("Total Bid Qty (top 5)", f"{int(total_bid)}")
    colb2.metric("Total Ask Qty (top 5)", f"{int(total_ask)}")
    colb3.metric(
        "Bid/Ask Qty Ratio",
        "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}",
    )
    colb4.metric("Top Bid / Ask", f"{_fmt_price(top_bid)} / {_fmt_price(top_ask)}")

    if footprint == "STRONG":
        st.error(f"Big Buyer Footprint (CE): STRONG – {desc}")
    elif footprint == "MILD":
        st.warning(f"Big Buyer Footprint (CE): MILD – {desc}")
    else:
        st.info(f"Big Buyer Footprint (CE): NONE – {desc}")


def layout_atm_pe_section(atm_pe_info, ob_info, nifty_change):
    st.subheader("🩸 ATM Put (PE) – Ramp-and-Dump Detector")

    if atm_pe_info is None:
        st.info("ATM NIFTY PE data not available yet.")
        return

    if nifty_change is None or pd.isna(nifty_change):
        st.info("NIFTY % change unavailable; cannot compute PE divergence.")
        return

    pe_chg = atm_pe_info["pct_change"]
    pe_ltp = atm_pe_info["ltp"]
    pe_symbol = atm_pe_info["tradingsymbol"]
    strike = atm_pe_info["strike"]
    expiry = atm_pe_info["expiry"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATM PE", f"{pe_symbol}")
    col2.metric("Strike", f"{int(strike)}")
    col3.metric("Expiry", str(expiry))
    col4.metric("PE LTP", _fmt_price(pe_ltp))

    col5, col6 = st.columns(2)
    col5.metric("PE % Change", _fmt_pct(pe_chg))
    col6.metric("NIFTY % Change", _fmt_pct(nifty_change))

    level = classify_pe_divergence(pe_chg, nifty_change)

    if level == "strong":
        note = (
            "NIFTY is strong, but ATM PE is flat or rising – "
            "possible aggressive put accumulation into the ramp."
        )
        st.error(f"Put Divergence: STRONG – {note}")
    elif level == "mild":
        note = (
            "NIFTY is strong, but ATM PE is not collapsing – "
            "mild sign of cautious put buying / hedging."
        )
        st.warning(f"Put Divergence: MILD – {note}")
    else:
        note = "No special PE divergence detected."
        st.info(f"Put Divergence: NEUTRAL – {note}")

    st.markdown("**Order Book Footprint (ATM PE)**")

    if ob_info is None:
        st.info("Order book data for PE not available.")
        return

    total_bid = ob_info["total_bid_qty"]
    total_ask = ob_info["total_ask_qty"]
    ratio = ob_info["bid_dom_ratio"]
    footprint = ob_info["footprint"]
    desc = ob_info["description"]
    top_bid = ob_info["top_bid_price"]
    top_ask = ob_info["top_ask_price"]

    colb1, colb2, colb3, colb4 = st.columns(4)
    colb1.metric("Total Bid Qty (top 5)", f"{int(total_bid)}")
    colb2.metric("Total Ask Qty (top 5)", f"{int(total_ask)}")
    colb3.metric(
        "Bid/Ask Qty Ratio",
        "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}",
    )
    colb4.metric("Top Bid / Ask", f"{_fmt_price(top_bid)} / {_fmt_price(top_ask)}")

    if footprint == "STRONG":
        st.error(f"Big Buyer Footprint (PE): STRONG – {desc}")
    elif footprint == "MILD":
        st.warning(f"Big Buyer Footprint (PE): MILD – {desc}")
    else:
        st.info(f"Big Buyer Footprint (PE): NONE – {desc}")


def layout_snapshot(df: pd.DataFrame, atm_ce_info, ob_ce_info, atm_pe_info, ob_pe_info):
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

    # Heavyweight vs index: shows both suppression & inflation flags
    layout_suppression_section(df)

    # Tabs for CE and PE to avoid clutter
    tab_ce, tab_pe = st.tabs(["ATM CE – Dip Buying", "ATM PE – Ramp & Dump"])

    with tab_ce:
        layout_atm_ce_section(atm_ce_info, ob_ce_info, nifty_change)

    with tab_pe:
        layout_atm_pe_section(atm_pe_info, ob_pe_info, nifty_change)

    st.subheader("🏋️ Heavyweights Snapshot")
    _render_heavyweights_table(df)


def _render_heavyweights_table(df: pd.DataFrame):
    display_df = df.copy()

    display_df["LTP"] = display_df["LTP"].map(_fmt_price)
    display_df["Prev Close"] = display_df["Prev Close"].map(_fmt_price)
    display_df["% Change"] = display_df["% Change"].map(_fmt_pct)

    st.dataframe(
        display_df[["Symbol", "LTP", "Prev Close", "% Change"]],
        use_container_width=True,
        hide_index=True,
    )


# --------- BURST MODE DECISION LOGIC ---------
def detect_strong_signal(
    df: pd.DataFrame,
    stats: dict | None,
    atm_ce_info: dict | None,
    ob_ce_info: dict | None,
    atm_pe_info: dict | None,
    ob_pe_info: dict | None,
) -> bool:
    """
    Decide if Burst Mode should be active.
    Criteria:
      - Suppression HIGH or Inflation HIGH
      - OR strong CE divergence + CE footprint MILD/STRONG
      - OR strong PE divergence + PE footprint MILD/STRONG
    """
    if df is None or df.empty:
        return False

    strong = False

    # 1) Suppression / Inflation
    if stats is not None:
        if stats.get("supp_label") == "HIGH" or stats.get("infl_label") == "HIGH":
            strong = True

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return strong

    nifty_change = nifty_rows.iloc[0]["% Change"]

    # 2) CE divergence + footprint
    if atm_ce_info is not None:
        ce_level = classify_ce_divergence(atm_ce_info.get("pct_change"), nifty_change)
        ce_fp = (ob_ce_info or {}).get("footprint", "NONE")
        if ce_level == "strong" and ce_fp in ("MILD", "STRONG"):
            strong = True

    # 3) PE divergence + footprint
    if atm_pe_info is not None:
        pe_level = classify_pe_divergence(atm_pe_info.get("pct_change"), nifty_change)
        pe_fp = (ob_pe_info or {}).get("footprint", "NONE")
        if pe_level == "strong" and pe_fp in ("MILD", "STRONG"):
            strong = True

    return strong


# --------- MAIN APP ---------
def main():
    layout_header()

    with st.sidebar:
        st.header("Settings")
        base_refresh_seconds = st.slider(
            "Base auto-refresh (seconds)", 5, 60, 15, step=5
        )
        st.caption(
            "When calm: uses this base interval.\n"
            f"When Burst Mode triggers: auto-switches to ~{BURST_REFRESH_SECONDS}s."
        )

    kite = get_kite_client()
    nifty_opt_df = get_nifty_option_instruments()

    # --------- FETCH + RENDER + DETECT STRONG SIGNAL ---------
    def run_fetch_and_render():
        df = fetch_ltp_snapshot(kite)
        if df.empty:
            st.warning("No LTP data available from Kite.")
            return False  # no burst

        nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]

        atm_ce_info = None
        atm_pe_info = None
        ob_ce_info = None
        ob_pe_info = None

        if not nifty_rows.empty and not nifty_opt_df.empty:
            nifty_spot = nifty_rows.iloc[0]["LTP"]
            if nifty_spot is not None and not pd.isna(nifty_spot):
                # ATM CE
                atm_ce_info = fetch_atm_option_quote(
                    kite, nifty_opt_df, float(nifty_spot), "CE"
                )
                if atm_ce_info is not None:
                    ob_ce_info = fetch_orderbook_for_option(kite, atm_ce_info)

                # ATM PE
                atm_pe_info = fetch_atm_option_quote(
                    kite, nifty_opt_df, float(nifty_spot), "PE"
                )
                if atm_pe_info is not None:
                    ob_pe_info = fetch_orderbook_for_option(kite, atm_pe_info)

        # Draw full layout
        layout_snapshot(df, atm_ce_info, ob_ce_info, atm_pe_info, ob_pe_info)

        # Compute stats for burst decision (separately from UI)
        stats = compute_suppression_stats(df)
        strong_signal = detect_strong_signal(
            df, stats, atm_ce_info, ob_ce_info, atm_pe_info, ob_pe_info
        )

        return strong_signal

    strong_signal = run_fetch_and_render()

    # --------- DECIDE EFFECTIVE REFRESH INTERVAL ---------
    if strong_signal:
        effective_refresh = BURST_REFRESH_SECONDS
        with st.sidebar:
            st.warning(
                f"🔥 Burst Mode ACTIVE – strong operator footprint detected.\n"
                f"Auto-refresh ~every {BURST_REFRESH_SECONDS} seconds."
            )
    else:
        effective_refresh = base_refresh_seconds
        with st.sidebar:
            st.info(
                f"Market calm (by this model). Using base refresh: "
                f"{base_refresh_seconds} seconds."
            )

    # --------- SCHEDULE NEXT AUTO-REFRESH ---------
    st_autorefresh(interval=int(effective_refresh * 1000), key="auto_refresh")


if __name__ == "__main__":
    main()
