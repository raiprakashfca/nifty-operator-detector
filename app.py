import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo

# --------- PAGE CONFIG ---------
st.set_page_config(
    page_title="NIFTY Operator Detector – Audio",
    layout="wide",
)

# --------- BURST MODE CONFIG ---------
BURST_REFRESH_SECONDS = 2  # when strong operator footprint is detected

# --------- EMBEDDED BEEP (WAV, BASE64) ---------
# Short 0.25s beep tone encoded as base64 WAV.
BEEP_BASE64 = """
UklGRkZWAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YSJWAAAAANAzz
zvHPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zv
TPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvP
PPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7H
PcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPd
M+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPM
c87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE
80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1
TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87
zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80z
vTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1Tzv
PPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7
HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTP
dM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPP
Mc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPc
E80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+
1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc8
7zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80
zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1Tz
vPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz
7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvT
PdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPP
PMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HP
cE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM
+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc
87zz7H
"""

def play_beep():
    """Inject a small audio tag to play a beep once."""
    b64 = "".join(BEEP_BASE64.split())  # remove newlines/spaces
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def trigger_high_divergence_beep(supp_label: str, infl_label: str):
    """
    Play beep on transition to HIGH (either suppression or inflation).
    Uses session_state so you don't get spammed on every refresh while it stays HIGH.
    """
    key = "high_divergence_active"
    prev = st.session_state.get(key, False)
    current = (supp_label == "HIGH") or (infl_label == "HIGH")

    if current and not prev:
        play_beep()  # rising edge: NORMAL/MILD -> HIGH

    st.session_state[key] = current


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

NIFTY_INDEX_SYMBOL = "NIFTY 50"   # NSE index


# --------- KITE CLIENT HELPERS ---------
@st.cache_resource(show_spinner=False)
def get_kite_client() -> KiteConnect:
    """Create and cache a KiteConnect client using secrets."""
    try:
        api_key = st.secrets["kite"]["api_key"]
        access_token = st.secrets["kite"]["access_token"]
    except Exception:
        st.error(
            "Kite API credentials not found in Streamlit secrets. "
            "Add them as:\n\n"
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
    """
    kite = get_kite_client()
    try:
        instruments = kite.instruments("NFO")
    except Exception as e:
        st.error(f"Error fetching NFO instruments from Kite: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(instruments)
    if df.empty:
        return df

    df = df[
        (df["exchange"] == "NFO")
        & (df["segment"] == "NFO-OPT")
        & (df["name"] == "NIFTY")
    ].copy()

    if df.empty:
        return df

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["strike"] = df["strike"].astype(float)
    return df


def build_instrument_list():
    """Build the list of instruments to query via ohlc()."""
    instruments = [f"NSE:{sym}" for sym in HEAVYWEIGHT_SYMBOLS]
    instruments.append(f"NSE:{NIFTY_INDEX_SYMBOL}")
    return instruments


def ensure_nifty_change(df: pd.DataFrame, kite: KiteConnect) -> pd.DataFrame:
    """
    If NIFTY % Change is missing, try a fallback quote() call to compute it.
    Silent if it fails; we just keep %Change as NaN.
    """
    if df is None or df.empty:
        return df

    mask = df["Symbol"].eq(NIFTY_INDEX_SYMBOL)
    if not mask.any():
        return df

    idx = df[mask].index[0]
    pct = df.at[idx, "% Change"]

    if pd.isna(pct):
        try:
            q = kite.quote([f"NSE:{NIFTY_INDEX_SYMBOL}"])
            data = list(q.values())[0]
            last_price = data.get("last_price")
            ohlc = data.get("ohlc", {}) or {}
            close = ohlc.get("close")

            if last_price is not None and close not in (None, 0):
                df.at[idx, "LTP"] = last_price
                df.at[idx, "Prev Close"] = close
                df.at[idx, "% Change"] = ((last_price - close) / close) * 100.0
        except Exception:
            pass  # stay quiet

    return df


def fetch_ltp_snapshot(kite: KiteConnect) -> pd.DataFrame:
    """
    Fetch LTP + previous close for NIFTY and heavyweights using kite.ohlc().
    """
    instruments = build_instrument_list()

    try:
        ohlc_data = kite.ohlc(instruments)
    except Exception as e:
        st.error(f"Error fetching OHLC data from Kite: {e}")
        return pd.DataFrame()

    rows = []
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    for instrument in instruments:
        data = ohlc_data.get(instrument, {}) or {}
        try:
            _, symbol = instrument.split(":", 1)
        except ValueError:
            symbol = instrument

        last_price = data.get("last_price")
        ohlc = data.get("ohlc", {}) or {}
        prev_close = ohlc.get("close")

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

    df = ensure_nifty_change(df, kite)

    df["is_nifty"] = df["Symbol"].eq(NIFTY_INDEX_SYMBOL)
    df = df.sort_values(
        by=["is_nifty", "% Change"],
        ascending=[False, False],
        ignore_index=True,
    ).drop(columns=["is_nifty"])

    return df


# --------- SUPPRESSION / INFLATION LOGIC (NIFTY-based) ---------
def compute_suppression_stats(df: pd.DataFrame):
    """
    Compute (NIFTY-based):
      - Nifty % change
      - Avg heavyweights % change
      - Divergence (heavy - Nifty)
      - Suppression label
      - Inflation label
    """
    if df is None or df.empty:
        return None

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return None

    nifty_row = nifty_rows.iloc[0]

    # heavyweights: everything except NIFTY
    heavy_df = df[df["Symbol"] != NIFTY_INDEX_SYMBOL]
    if heavy_df.empty:
        return None

    nifty_change = nifty_row["% Change"]
    avg_heavy_change = heavy_df["% Change"].mean()

    if pd.isna(nifty_change) or pd.isna(avg_heavy_change):
        return None

    divergence = avg_heavy_change - nifty_change

    suppression_label = "NORMAL"
    suppression_explanation = "No strong suppression detected."
    inflation_label = "NORMAL"
    inflation_explanation = "No strong inflation detected."

    abs_div = abs(divergence)

    # Suppression: NIFTY mildly down, heavyweights materially weaker
    if (
        nifty_change <= -0.20
        and nifty_change >= -1.50
        and divergence <= -0.30
    ):
        if abs_div >= 1.0:
            suppression_label = "HIGH"
            suppression_explanation = (
                "Heavyweights much weaker than NIFTY – strong suppression risk."
            )
        else:
            suppression_label = "MILD"
            suppression_explanation = (
                "Heavyweights weaker than NIFTY – some suppression."
            )

    # Inflation: NIFTY mildly up, heavyweights materially stronger
    if (
        nifty_change >= 0.20
        and nifty_change <= 1.50
        and divergence >= 0.30
    ):
        if abs_div >= 1.0:
            inflation_label = "HIGH"
            inflation_explanation = (
                "Heavyweights much stronger than NIFTY – strong inflation risk."
            )
        else:
            inflation_label = "MILD"
            inflation_explanation = (
                "Heavyweights stronger than NIFTY – some inflation."
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


# --------- DEEP ITM (>=100pts) OPTION FINDERS & QUOTES ---------
def find_itm_near_spot_instrument(
    nifty_opt_df: pd.DataFrame, nifty_spot: float, option_type: str
):
    """
    Find ITM option at least 100 points in-the-money, closest to SPOT:

      - For CE: strike <= spot - 100 (deep ITM), choose closest to spot.
      - For PE: strike >= spot + 100 (deep ITM), choose closest to spot.

    If no such deep ITM exists:
      1) Fall back to strict ITM by spot (CE: strike < spot, PE: strike > spot).
      2) If even that is empty, fall back to nearest strike overall (incl. ATM).
    """
    if nifty_opt_df is None or nifty_opt_df.empty:
        return None
    if nifty_spot is None or pd.isna(nifty_spot):
        return None

    option_type = option_type.upper()
    if option_type not in ("CE", "PE"):
        return None

    today = date.today()

    df = nifty_opt_df[
        (nifty_opt_df["instrument_type"] == option_type)
        & (nifty_opt_df["expiry"] >= today)
    ].copy()
    if df.empty:
        return None

    # Step 1: Deep ITM (>= 100 points)
    if option_type == "CE":
        df_deep_itm = df[df["strike"] <= (nifty_spot - 100)].copy()
    else:  # PE
        df_deep_itm = df[df["strike"] >= (nifty_spot + 100)].copy()

    if not df_deep_itm.empty:
        df_deep_itm["spot_diff"] = (df_deep_itm["strike"] - nifty_spot).abs()
        df_sel = df_deep_itm
    else:
        # Step 2: strict ITM (no 100pt buffer, but still not ATM)
        if option_type == "CE":
            df_itm = df[df["strike"] < nifty_spot].copy()
        else:  # PE
            df_itm = df[df["strike"] > nifty_spot].copy()

        if not df_itm.empty:
            df_itm["spot_diff"] = (df_itm["strike"] - nifty_spot).abs()
            df_sel = df_itm
        else:
            # Step 3: fallback – nearest to spot including ATM
            df["spot_diff"] = (df["strike"] - nifty_spot).abs()
            df_sel = df

    df_sel = df_sel.sort_values(["spot_diff", "expiry"])
    return df_sel.iloc[0]


def fetch_itm_option_quote(
    kite: KiteConnect, nifty_opt_df: pd.DataFrame, nifty_spot: float, option_type: str
):
    """
    Find deep-ITM-near-spot option (CE or PE) and fetch:
      - LTP, % change, prev close
      - cumulative volume
      - last traded quantity (LTQ)
    Uses kite.quote() so we also get volume and LTQ.
    """
    row = find_itm_near_spot_instrument(nifty_opt_df, nifty_spot, option_type)
    if row is None:
        return None

    tradingsymbol = row["tradingsymbol"]
    strike = row["strike"]
    expiry = row["expiry"]
    instrument = f"NFO:{tradingsymbol}"

    try:
        q = kite.quote([instrument])
    except Exception as e:
        st.error(f"Error fetching ITM {option_type} quote from Kite: {e}")
        return None

    if not q:
        return None

    data = list(q.values())[0]
    last_price = data.get("last_price")
    last_qty = data.get("last_quantity")  # LTQ
    ohlc = data.get("ohlc", {}) or {}
    prev_close = ohlc.get("close")

    # volume key is 'volume_traded'; also try 'volume'
    volume_total = data.get("volume_traded")
    if volume_total is None:
        volume_total = data.get("volume")

    if last_price is not None and prev_close not in (None, 0):
        pct_change = ((last_price - prev_close) / prev_close) * 100.0
    else:
        pct_change = None

    return {
        "tradingsymbol": tradingsymbol,
        "strike": float(strike),
        "expiry": expiry,
        "ltp": last_price,
        "last_quantity": last_qty,
        "prev_close": prev_close,
        "pct_change": pct_change,
        "instrument": instrument,
        "option_type": option_type.upper(),
        "volume_total": volume_total,
    }


# --------- ORDER BOOK / DEPTH LOGIC ---------
def fetch_orderbook_for_option(kite: KiteConnect, opt_info: dict):
    """Fetch depth for an option and compute bid/ask dominance."""
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

    if bid_dom_ratio >= 2.0 and total_bid_qty > 0:
        footprint = "STRONG"
        desc = "Bid side heavily stacked vs ask."
    elif bid_dom_ratio >= 1.2 and total_bid_qty > 0:
        footprint = "MILD"
        desc = "Bid side mildly dominant."
    else:
        footprint = "NONE"
        desc = "No clear bid dominance."

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


def _fmt_int(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{int(x)}"
    except Exception:
        return "-"


# --------- DIVERGENCE CLASSIFIERS ---------
def classify_ce_divergence(ce_chg, nifty_change) -> str:
    """Return 'strong', 'mild', or 'neutral' for CE divergence."""
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
    """Return 'strong', 'mild', or 'neutral' for PE divergence."""
    if pe_chg is None or pd.isna(pe_chg):
        return "neutral"
    if nifty_change is None or pd.isna(nifty_change):
        return "neutral"

    pe = float(pe_chg)
    nf = float(nifty_change)

    if nf >= 0.20:  # Nifty strong
        if pe >= 0.0:
            return "strong"
        elif pe > -3.0:
            return "mild"
    return "neutral"


# --------- RECENT VOLUME (15s ESTIMATE) ---------
def compute_recent_volume_15s(instrument: str, current_volume):
    """
    Approximate volume in last 15 seconds using cumulative volume deltas.

    - Stores (volume, timestamp) per instrument in st.session_state.
    - On each refresh, compute delta + elapsed seconds.
    - Scale delta to a 15-second equivalent: delta * (15 / elapsed).
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    if current_volume is None or pd.isna(current_volume):
        st.session_state[f"vol_state_{instrument}"] = (None, now)
        return None

    key = f"vol_state_{instrument}"
    last = st.session_state.get(key)

    if last is None:
        st.session_state[key] = (current_volume, now)
        return None

    last_vol, last_time = last
    st.session_state[key] = (current_volume, now)

    if last_vol is None or last_time is None:
        return None

    delta = current_volume - last_vol
    if delta <= 0:
        return None

    elapsed = (now - last_time).total_seconds()
    if elapsed <= 0:
        return None

    est_15s = delta * (15.0 / elapsed)
    return max(est_15s, 0.0)


# --------- LAYOUT HELPERS ---------
def layout_header():
    st.title("NIFTY Operator Detector – Burst Mode + Audio")
    st.caption(
        "NIFTY vs heavyweights (NSE) + ≥100-pt ITM CE/PE + LTP/LTQ + order book + est. 15s volume.\n"
        "Audio alert on HIGH divergence. CE and PE visible together. "
        "Burst Mode speeds up refresh on strong footprints."
    )


def layout_suppression_section(df: pd.DataFrame):
    stats = compute_suppression_stats(df)
    st.subheader("🧲 NIFTY vs Heavyweights")

    if stats is None:
        st.info("Not enough clean data yet for suppression/inflation.")
        return

    nifty_chg = stats["nifty_change"]
    heavy_chg = stats["avg_heavy_change"]
    divergence = stats["divergence"]
    supp_label = stats["supp_label"]
    supp_expl = stats["supp_expl"]
    infl_label = stats["infl_label"]
    infl_expl = stats["infl_expl"]

    # Trigger audio if we just flipped to HIGH
    trigger_high_divergence_beep(supp_label, infl_label)

    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY %", _fmt_pct(nifty_chg))
    col2.metric("Avg Heavyweights %", _fmt_pct(heavy_chg))
    col3.metric("Heavy - NIFTY", _fmt_pct(divergence))

    if supp_label == "HIGH":
        st.error(f"Suppression: HIGH – {supp_expl}")
    elif supp_label == "MILD":
        st.warning(f"Suppression: MILD – {supp_expl}")
    else:
        st.info(f"Suppression: NORMAL – {supp_expl}")

    if infl_label == "HIGH":
        st.error(f"Inflation: HIGH – {infl_expl}")
    elif infl_label == "MILD":
        st.warning(f"Inflation: MILD – {infl_expl}")
    else:
        st.info(f"Inflation: NORMAL – {infl_expl}")


def layout_itm_ce_section(itm_ce_info, ob_info, nifty_change):
    st.subheader("🎯 ≥100pt ITM CE – Dip Buying")

    if itm_ce_info is None:
        st.info("Deep ITM NIFTY CE not available.")
        return

    ce_chg = itm_ce_info["pct_change"]
    ce_ltp = itm_ce_info["ltp"]
    ce_ltq = itm_ce_info.get("last_quantity")
    ce_symbol = itm_ce_info["tradingsymbol"]
    strike = itm_ce_info["strike"]
    expiry = itm_ce_info["expiry"]
    volume_total = itm_ce_info.get("volume_total")
    instrument = itm_ce_info["instrument"]

    est_vol_15s = compute_recent_volume_15s(instrument, volume_total)
    strike_label = int(strike)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ITM CE", ce_symbol)
        st.metric("Strike", f"{strike_label}")
    with col2:
        st.metric("Expiry", str(expiry))

    # LTP + LTQ row
    col_ltp, col_ltq = st.columns(2)
    col_ltp.metric("CE LTP", _fmt_price(ce_ltp))
    col_ltq.metric("CE LTQ", _fmt_int(ce_ltq))

    col3, col4, col5 = st.columns(3)
    col3.metric("CE %", _fmt_pct(ce_chg))
    col4.metric("NIFTY %", _fmt_pct(nifty_change))
    col5.metric(
        "Est Vol (15s)",
        "-" if est_vol_15s is None else _fmt_int(est_vol_15s),
    )

    level = classify_ce_divergence(ce_chg, nifty_change)

    if level == "strong":
        msg = "NIFTY weak, deep ITM CE flat/green – strong call accumulation risk."
        st.error(f"Divergence (Strike {strike_label}): STRONG – {msg}")
    elif level == "mild":
        msg = "NIFTY weak, deep ITM CE relatively strong – mild call buying."
        st.warning(f"Divergence (Strike {strike_label}): MILD – {msg}")
    else:
        st.info(f"Divergence (Strike {strike_label}): NEUTRAL – no special CE signal.")

    st.markdown("**Order Book (ITM CE)**")

    if ob_info is None:
        st.info("Order book for CE not available.")
        return

    total_bid = ob_info["total_bid_qty"]
    total_ask = ob_info["total_ask_qty"]
    ratio = ob_info["bid_dom_ratio"]
    footprint = ob_info["footprint"]
    desc = ob_info["description"]
    top_bid = ob_info["top_bid_price"]
    top_ask = ob_info["top_ask_price"]

    # First row: quantities & ratio
    colb1, colb2, colb3 = st.columns(3)
    colb1.metric("Bid Qty (top 5)", _fmt_int(total_bid))
    colb2.metric("Ask Qty (top 5)", _fmt_int(total_ask))
    colb3.metric(
        "Bid/Ask Qty",
        "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}",
    )

    # Second row: prices
    colp1, colp2 = st.columns(2)
    colp1.metric("Top Bid", _fmt_price(top_bid))
    colp2.metric("Top Ask", _fmt_price(top_ask))

    if footprint == "STRONG":
        st.error(f"Footprint (Strike {strike_label}): STRONG – {desc}")
    elif footprint == "MILD":
        st.warning(f"Footprint (Strike {strike_label}): MILD – {desc}")
    else:
        st.info(f"Footprint (Strike {strike_label}): NONE – {desc}")


def layout_itm_pe_section(itm_pe_info, ob_info, nifty_change):
    st.subheader("🩸 ≥100pt ITM PE – Ramp & Dump")

    if itm_pe_info is None:
        st.info("Deep ITM NIFTY PE not available.")
        return

    pe_chg = itm_pe_info["pct_change"]
    pe_ltp = itm_pe_info["ltp"]
    pe_ltq = itm_pe_info.get("last_quantity")
    pe_symbol = itm_pe_info["tradingsymbol"]
    strike = itm_pe_info["strike"]
    expiry = itm_pe_info["expiry"]
    volume_total = itm_pe_info.get("volume_total")
    instrument = itm_pe_info["instrument"]

    est_vol_15s = compute_recent_volume_15s(instrument, volume_total)
    strike_label = int(strike)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ITM PE", pe_symbol)
        st.metric("Strike", f"{strike_label}")
    with col2:
        st.metric("Expiry", str(expiry))

    # LTP + LTQ row
    col_ltp, col_ltq = st.columns(2)
    col_ltp.metric("PE LTP", _fmt_price(pe_ltp))
    col_ltq.metric("PE LTQ", _fmt_int(pe_ltq))

    col3, col4, col5 = st.columns(3)
    col3.metric("PE %", _fmt_pct(pe_chg))
    col4.metric("NIFTY %", _fmt_pct(nifty_change))
    col5.metric(
        "Est Vol (15s)",
        "-" if est_vol_15s is None else _fmt_int(est_vol_15s),
    )

    level = classify_pe_divergence(pe_chg, nifty_change)

    if level == "strong":
        msg = "NIFTY strong, deep ITM PE flat/green – strong put accumulation risk."
        st.error(f"Divergence (Strike {strike_label}): STRONG – {msg}")
    elif level == "mild":
        msg = "NIFTY strong, deep ITM PE not collapsing – mild put buying/hedging."
        st.warning(f"Divergence (Strike {strike_label}): MILD – {msg}")
    else:
        st.info(f"Divergence (Strike {strike_label}): NEUTRAL – no special PE signal.")

    st.markdown("**Order Book (ITM PE)**")

    if ob_info is None:
        st.info("Order book for PE not available.")
        return

    total_bid = ob_info["total_bid_qty"]
    total_ask = ob_info["total_ask_qty"]
    ratio = ob_info["bid_dom_ratio"]
    footprint = ob_info["footprint"]
    desc = ob_info["description"]
    top_bid = ob_info["top_bid_price"]
    top_ask = ob_info["top_ask_price"]

    # First row: quantities & ratio
    colb1, colb2, colb3 = st.columns(3)
    colb1.metric("Bid Qty (top 5)", _fmt_int(total_bid))
    colb2.metric("Ask Qty (top 5)", _fmt_int(total_ask))
    colb3.metric(
        "Bid/Ask Qty",
        "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}",
    )

    # Second row: prices
    colp1, colp2 = st.columns(2)
    colp1.metric("Top Bid", _fmt_price(top_bid))
    colp2.metric("Top Ask", _fmt_price(top_ask))

    if footprint == "STRONG":
        st.error(f"Footprint (Strike {strike_label}): STRONG – {desc}")
    elif footprint == "MILD":
        st.warning(f"Footprint (Strike {strike_label}): MILD – {desc}")
    else:
        st.info(f"Footprint (Strike {strike_label}): NONE – {desc}")


def layout_snapshot(df: pd.DataFrame, itm_ce_info, ob_ce_info, itm_pe_info, ob_pe_info):
    if df is None or df.empty:
        st.warning("No data returned from Kite.")
        return

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        st.warning("NIFTY row missing. Showing only heavyweights.")
        _render_heavyweights_table(df)
        return

    nifty_row = nifty_rows.iloc[0]
    nifty_ltp = nifty_row["LTP"]
    nifty_change = nifty_row["% Change"]
    nifty_ts = nifty_row["Timestamp"]

    nifty_ltp_display = _fmt_price(nifty_ltp)
    nifty_change_display = _fmt_pct(nifty_change)

    if isinstance(nifty_ts, datetime):
        ts_display = nifty_ts.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
    else:
        ts_display = str(nifty_ts)

    st.subheader("📈 NIFTY Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("NIFTY LTP", nifty_ltp_display)
    col2.metric("NIFTY %", nifty_change_display)
    col3.write(f"Timestamp (IST): {ts_display}")

    layout_suppression_section(df)

    st.subheader("🎯 Options Operator Footprint – Deep ITM CE & PE")
    col_ce, col_pe = st.columns(2)
    with col_ce:
        layout_itm_ce_section(itm_ce_info, ob_ce_info, nifty_change)
    with col_pe:
        layout_itm_pe_section(itm_pe_info, ob_pe_info, nifty_change)

    st.subheader("🏋️ Heavyweights")
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
    itm_ce_info: dict | None,
    ob_ce_info: dict | None,
    itm_pe_info: dict | None,
    ob_pe_info: dict | None,
) -> bool:
    """
    Burst Mode when:
      - Suppression HIGH or Inflation HIGH
      - OR strong CE divergence + CE footprint MILD/STRONG
      - OR strong PE divergence + PE footprint MILD/STRONG
    """
    if df is None or df.empty:
        return False

    strong = False

    if stats is not None:
        if stats.get("supp_label") == "HIGH" or stats.get("infl_label") == "HIGH":
            strong = True

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return strong

    nifty_change = nifty_rows.iloc[0]["% Change"]

    if itm_ce_info is not None:
        ce_level = classify_ce_divergence(itm_ce_info.get("pct_change"), nifty_change)
        ce_fp = (ob_ce_info or {}).get("footprint", "NONE")
        if ce_level == "strong" and ce_fp in ("MILD", "STRONG"):
            strong = True

    if itm_pe_info is not None:
        pe_level = classify_pe_divergence(itm_pe_info.get("pct_change"), nifty_change)
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
            "Base auto-refresh (s)", 5, 60, 15, step=5
        )
        st.caption(
            "Calm market → base interval.\n"
            f"Strong signal → auto {BURST_REFRESH_SECONDS}s Burst Mode."
        )

    kite = get_kite_client()
    nifty_opt_df = get_nifty_option_instruments()

    def run_fetch_and_render():
        df = fetch_ltp_snapshot(kite)
        if df.empty:
            st.warning("No LTP/OHLC data from Kite.")
            return False

        nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]

        itm_ce_info = None
        itm_pe_info = None
        ob_ce_info = None
        ob_pe_info = None

        if not nifty_rows.empty and not nifty_opt_df.empty:
            nifty_spot = nifty_rows.iloc[0]["LTP"]
            if nifty_spot is not None and not pd.isna(nifty_spot):
                # Deep ITM CE
                itm_ce_info = fetch_itm_option_quote(
                    kite, nifty_opt_df, float(nifty_spot), "CE"
                )
                if itm_ce_info is not None:
                    ob_ce_info = fetch_orderbook_for_option(kite, itm_ce_info)

                # Deep ITM PE
                itm_pe_info = fetch_itm_option_quote(
                    kite, nifty_opt_df, float(nifty_spot), "PE"
                )
                if itm_pe_info is not None:
                    ob_pe_info = fetch_orderbook_for_option(kite, itm_pe_info)

        layout_snapshot(df, itm_ce_info, ob_ce_info, itm_pe_info, ob_pe_info)

        stats = compute_suppression_stats(df)
        strong_signal = detect_strong_signal(
            df, stats, itm_ce_info, ob_ce_info, itm_pe_info, ob_pe_info
        )
        return strong_signal

    strong_signal = run_fetch_and_render()

    if strong_signal:
        effective_refresh = BURST_REFRESH_SECONDS
        with st.sidebar:
            st.warning(
                f"🔥 Burst Mode ACTIVE – strong footprint detected.\n"
                f"Refresh ~every {BURST_REFRESH_SECONDS}s."
            )
    else:
        effective_refresh = base_refresh_seconds
        with st.sidebar:
            st.info(
                f"Market calm (by this model). "
                f"Base refresh: {base_refresh_seconds}s."
            )

    st_autorefresh(interval=int(effective_refresh * 1000), key="auto_refresh")


if __name__ == "__main__":
    main()
