import streamlit as st
from streamlit_autorefresh import st_autorefresh

from kiteconnect import KiteConnect
import gspread
import pandas as pd

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs

# =========================
# PAGE CONFIG
# =========================
APP_TITLE = "NIFTY Operator Detector"
TZ = ZoneInfo("Asia/Kolkata")

st.set_page_config(page_title=APP_TITLE, layout="wide")

# =========================
# DEFAULTS
# =========================
DEFAULT_HEAVYWEIGHTS = [
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

NIFTY_INDEX_SYMBOL = "NIFTY 50"  # NSE index symbol on Kite for quote/ohlc

DEFAULT_HISTORY_WINDOW_MINUTES = 5
DEFAULT_BASE_REFRESH_SECONDS = 15
BURST_REFRESH_SECONDS = 2

# =========================
# AUDIO BEEP (BASE64 WAV)
# =========================
BEEP_BASE64 = """
UklGRkZWAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YSJWAAAAANAzz
zzvHPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zvTPdM+1TzvPPPMc87zz7HPcE80zv
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
7zz7HPcE80zvTPdM+1TzvPPPMc87zz7H
"""

def _play_beep_once() -> None:
    b64 = "".join(BEEP_BASE64.split())
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )

def _beep_on_high_transition(is_high: bool, enabled: bool) -> None:
    if not enabled:
        st.session_state["_high_active"] = bool(is_high)
        return

    prev = bool(st.session_state.get("_high_active", False))
    if is_high and not prev:
        _play_beep_once()
    st.session_state["_high_active"] = bool(is_high)

# =========================
# GOOGLE SHEETS (TOKEN STORE)
# =========================
@st.cache_resource(show_spinner=False)
def _get_gspread_client():
    try:
        sa_info = st.secrets["gcp_service_account"]
    except Exception:
        st.error(
            "Missing Google service account JSON in Streamlit secrets.\n\n"
            "Add it as st.secrets['gcp_service_account'] (full JSON dict) and share "
            "the ZerodhaTokenStore sheet with the service account email."
        )
        st.stop()

    try:
        return gspread.service_account_from_dict(sa_info)
    except Exception as e:
        st.error(f"Failed to init gspread client: {e}")
        st.stop()

def _open_token_sheet():
    gc = _get_gspread_client()
    try:
        sh = gc.open("ZerodhaTokenStore")
        return sh.sheet1
    except Exception as e:
        st.error(
            "Could not open Google Sheet 'ZerodhaTokenStore'. Ensure it exists and is shared.\n\n"
            f"Details: {e}"
        )
        st.stop()

def read_zerodha_tokens_from_sheet():
    """Expected layout in Sheet1 row 1: A1=API Key, B1=API Secret, C1=Access Token."""
    ws = _open_token_sheet()
    try:
        row = ws.row_values(1)
    except Exception as e:
        st.error(f"Failed to read row 1 from ZerodhaTokenStore: {e}")
        st.stop()

    api_key = row[0].strip() if len(row) >= 1 else ""
    api_secret = row[1].strip() if len(row) >= 2 else ""
    access_token = row[2].strip() if len(row) >= 3 else ""
    return api_key, api_secret, access_token

def write_access_token_to_sheet(access_token: str) -> None:
    ws = _open_token_sheet()
    try:
        ws.update("C1", access_token)
    except Exception as e:
        st.error(f"Failed to write access token to ZerodhaTokenStore!C1: {e}")
        st.stop()

# =========================
# KITE AUTH (STREAMLIT-FRIENDLY)
# =========================
def _extract_request_token(pasted_url: str) -> str | None:
    if not pasted_url:
        return None
    try:
        parsed = urlparse(pasted_url.strip())
        qs = parse_qs(parsed.query)
        tok = (qs.get("request_token") or [None])[0]
        return tok
    except Exception:
        return None

def _auth_sidebar() -> tuple[str, str, str]:
    """Returns (api_key, api_secret, access_token). Also handles refresh flow."""
    st.sidebar.header("🔐 Zerodha Login")

    api_key, api_secret, access_token = read_zerodha_tokens_from_sheet()

    if not api_key:
        st.sidebar.error("API Key missing in ZerodhaTokenStore!A1")
        st.stop()

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    with st.sidebar.expander("Daily token refresh", expanded=False):
        st.markdown(
            "**How it works:** Click login → Zerodha redirects to your redirect URL with a `request_token`. "
            "Paste the redirected URL below to generate a fresh `access_token` and store it in Google Sheets."
        )
        st.link_button("Open Zerodha Login", login_url)

        pasted = st.text_input(
            "Paste redirected URL (contains request_token)",
            placeholder="https://your-redirect-uri?request_token=...",
            help="After login, copy the full redirected URL from the browser address bar and paste it here.",
        )
        req_tok = _extract_request_token(pasted)

        c1, c2 = st.columns(2)
        with c1:
            do_gen = st.button("Generate new access token", use_container_width=True, type="primary")
        with c2:
            do_clear = st.button("Clear caches", use_container_width=True)

        if do_clear:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success("Caches cleared.")

        if do_gen:
            if not api_secret:
                st.sidebar.error("API Secret missing in ZerodhaTokenStore!B1")
                st.stop()
            if not req_tok:
                st.sidebar.error("No request_token found in the pasted URL.")
                st.stop()

            try:
                sess = kite.generate_session(req_tok, api_secret=api_secret)
                new_access = sess.get("access_token")
            except Exception as e:
                st.sidebar.error(f"Token generation failed: {e}")
                st.stop()

            if not new_access:
                st.sidebar.error("No access_token returned by Zerodha.")
                st.stop()

            write_access_token_to_sheet(new_access)

            st.session_state["_forced_access_token"] = new_access
            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success("Access token updated in ZerodhaTokenStore!C1")
            st.rerun()

    access_token = st.session_state.get("_forced_access_token", access_token)

    if not access_token:
        st.sidebar.warning("Access token missing in ZerodhaTokenStore!C1 → refresh token.")

    return api_key, api_secret, access_token

@st.cache_resource(show_spinner=False)
def get_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    if access_token:
        kite.set_access_token(access_token)
    return kite

# =========================
# MARKET DATA
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def get_nifty_option_instruments(kite_api_key: str, kite_access_token: str) -> pd.DataFrame:
    """Cache NIFTY options instrument list for 1 hour."""
    kite = get_kite_client(kite_api_key, kite_access_token)
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    if df.empty:
        return df

    df = df[
        (df["exchange"] == "NFO")
        & (df["segment"] == "NFO-OPT")
        & (df["name"] == "NIFTY")
    ].copy()

    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["strike"] = df["strike"].astype(float)
    return df

def build_instrument_list(heavyweights: list[str]) -> list[str]:
    instruments = [f"NSE:{sym}" for sym in heavyweights]
    instruments.append(f"NSE:{NIFTY_INDEX_SYMBOL}")
    return instruments

def fetch_ltp_snapshot(kite: KiteConnect, heavyweights: list[str]) -> pd.DataFrame:
    """One OHLC call for NIFTY + heavyweights."""
    instruments = build_instrument_list(heavyweights)
    ohlc_data = kite.ohlc(instruments)
    now = datetime.now(TZ)

    rows = []
    for instrument in instruments:
        data = ohlc_data.get(instrument, {}) or {}
        _, symbol = instrument.split(":", 1)

        last_price = data.get("last_price")
        ohlc = data.get("ohlc", {}) or {}
        prev_close = ohlc.get("close")

        pct_change = None
        if last_price is not None and prev_close not in (None, 0):
            pct_change = ((last_price - prev_close) / prev_close) * 100.0

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

    df["_is_nifty"] = df["Symbol"].eq(NIFTY_INDEX_SYMBOL)
    df = df.sort_values(by=["_is_nifty", "% Change"], ascending=[False, False]).drop(columns=["_is_nifty"])
    df = df.reset_index(drop=True)
    return df

# =========================
# MODEL (SUPPRESSION / INFLATION)
# =========================
def compute_suppression_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return None

    nifty_change = nifty_rows.iloc[0]["% Change"]
    heavy_df = df[df["Symbol"] != NIFTY_INDEX_SYMBOL]
    if heavy_df.empty:
        return None

    avg_heavy_change = heavy_df["% Change"].mean()
    if pd.isna(nifty_change) or pd.isna(avg_heavy_change):
        return None

    divergence = float(avg_heavy_change - nifty_change)

    suppression_label = "NORMAL"
    inflation_label = "NORMAL"

    abs_div = abs(divergence)

    if nifty_change <= -0.20 and nifty_change >= -1.50 and divergence <= -0.30:
        suppression_label = "HIGH" if abs_div >= 1.0 else "MILD"

    if nifty_change >= 0.20 and nifty_change <= 1.50 and divergence >= 0.30:
        inflation_label = "HIGH" if abs_div >= 1.0 else "MILD"

    return {
        "nifty_change": float(nifty_change),
        "avg_heavy_change": float(avg_heavy_change),
        "divergence": float(divergence),
        "supp_label": suppression_label,
        "infl_label": inflation_label,
    }

# =========================
# DEEP ITM OPTION PICK + SINGLE QUOTE (CE+PE together)
# =========================
def find_itm_near_spot_instrument(nifty_opt_df: pd.DataFrame, nifty_spot: float, option_type: str):
    if nifty_opt_df is None or nifty_opt_df.empty or nifty_spot is None or pd.isna(nifty_spot):
        return None

    option_type = option_type.upper().strip()
    if option_type not in ("CE", "PE"):
        return None

    today = date.today()
    df = nifty_opt_df[(nifty_opt_df["instrument_type"] == option_type) & (nifty_opt_df["expiry"] >= today)].copy()
    if df.empty:
        return None

    if option_type == "CE":
        df1 = df[df["strike"] <= (nifty_spot - 100)].copy()
    else:
        df1 = df[df["strike"] >= (nifty_spot + 100)].copy()

    if df1.empty:
        if option_type == "CE":
            df1 = df[df["strike"] < nifty_spot].copy()
        else:
            df1 = df[df["strike"] > nifty_spot].copy()

    if df1.empty:
        df1 = df.copy()

    df1["spot_diff"] = (df1["strike"] - nifty_spot).abs()
    df1 = df1.sort_values(["spot_diff", "expiry"], ascending=[True, True])
    return df1.iloc[0]

def quote_deep_itm_pair(kite: KiteConnect, nifty_opt_df: pd.DataFrame, nifty_spot: float):
    """Returns (ce_info, pe_info). Uses a SINGLE kite.quote([CE, PE]) call incl. depth."""
    ce_row = find_itm_near_spot_instrument(nifty_opt_df, nifty_spot, "CE")
    pe_row = find_itm_near_spot_instrument(nifty_opt_df, nifty_spot, "PE")

    if ce_row is None and pe_row is None:
        return None, None

    instruments = []
    if ce_row is not None:
        instruments.append(f"NFO:{ce_row['tradingsymbol']}")
    if pe_row is not None:
        instruments.append(f"NFO:{pe_row['tradingsymbol']}")

    q = kite.quote(instruments)

    def _build(row, instrument):
        data = q.get(instrument, {}) or {}
        last_price = data.get("last_price")
        last_qty = data.get("last_quantity")
        ohlc = data.get("ohlc", {}) or {}
        prev_close = ohlc.get("close")
        volume_total = data.get("volume_traded") or data.get("volume")

        pct_change = None
        if last_price is not None and prev_close not in (None, 0):
            pct_change = ((last_price - prev_close) / prev_close) * 100.0

        depth = data.get("depth", {}) or {}
        buys = depth.get("buy", []) or []
        sells = depth.get("sell", []) or []

        bid_qty = sum(l.get("quantity", 0) for l in buys)
        ask_qty = sum(l.get("quantity", 0) for l in sells)
        ratio = (bid_qty / ask_qty) if ask_qty > 0 else (float("inf") if bid_qty > 0 else 0.0)

        if ratio >= 2.0 and bid_qty > 0:
            fp = "STRONG"
        elif ratio >= 1.2 and bid_qty > 0:
            fp = "MILD"
        else:
            fp = "NONE"

        return {
            "tradingsymbol": row["tradingsymbol"],
            "strike": float(row["strike"]),
            "expiry": row["expiry"],
            "instrument": instrument,
            "ltp": last_price,
            "last_quantity": last_qty,
            "prev_close": prev_close,
            "pct_change": pct_change,
            "volume_total": volume_total,
            "depth": {
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "bid_ask_ratio": ratio,
                "top_bid": buys[0].get("price") if buys else None,
                "top_ask": sells[0].get("price") if sells else None,
                "footprint": fp,
            },
        }

    ce_info = _build(ce_row, f"NFO:{ce_row['tradingsymbol']}") if ce_row is not None else None
    pe_info = _build(pe_row, f"NFO:{pe_row['tradingsymbol']}") if pe_row is not None else None
    return ce_info, pe_info

# =========================
# DIVERGENCE CLASSIFIERS (HEURISTICS)
# =========================
def classify_ce_divergence(ce_chg, nifty_change) -> str:
    if ce_chg is None or pd.isna(ce_chg) or nifty_change is None or pd.isna(nifty_change):
        return "neutral"

    ce = float(ce_chg)
    nf = float(nifty_change)

    if nf <= -0.20:
        if ce >= 0.0:
            return "strong"
        if ce > nf + 3.0:
            return "mild"

    return "neutral"

def classify_pe_divergence(pe_chg, nifty_change) -> str:
    if pe_chg is None or pd.isna(pe_chg) or nifty_change is None or pd.isna(nifty_change):
        return "neutral"

    pe = float(pe_chg)
    nf = float(nifty_change)

    if nf >= 0.20:
        if pe >= 0.0:
            return "strong"
        if pe > -3.0:
            return "mild"

    return "neutral"

# =========================
# RECENT VOLUME (CUMULATIVE DELTA -> 15s estimate)
# =========================
def compute_recent_volume_15s(instrument: str, current_volume):
    now = datetime.now(TZ)

    if current_volume is None or pd.isna(current_volume):
        st.session_state[f"_vol_{instrument}"] = (None, now)
        return None

    last = st.session_state.get(f"_vol_{instrument}")
    st.session_state[f"_vol_{instrument}"] = (current_volume, now)

    if not last:
        return None

    last_vol, last_ts = last
    if last_vol is None or last_ts is None:
        return None

    delta = current_volume - last_vol
    if delta <= 0:
        return None

    elapsed = (now - last_ts).total_seconds()
    if elapsed <= 0:
        return None

    return max(delta * (15.0 / elapsed), 0.0)

# =========================
# HISTORY (IN-MEM)
# =========================
def update_history(history_window_minutes: int, df: pd.DataFrame, stats: dict | None, ce: dict | None, pe: dict | None):
    if df is None or df.empty:
        return

    nifty_rows = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nifty_rows.empty:
        return

    nifty_row = nifty_rows.iloc[0]
    ts: datetime = nifty_row["Timestamp"]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=TZ)

    record = {
        "_ts": ts,
        "Time": ts.astimezone(TZ).strftime("%H:%M:%S"),
        "NIFTY %": nifty_row["% Change"],
        "Supp": (stats or {}).get("supp_label"),
        "Infl": (stats or {}).get("infl_label"),
        "CE Strike": int(ce["strike"]) if ce else None,
        "CE %": ce.get("pct_change") if ce else None,
        "CE FP": (ce.get("depth") or {}).get("footprint") if ce else None,
        "PE Strike": int(pe["strike"]) if pe else None,
        "PE %": pe.get("pct_change") if pe else None,
        "PE FP": (pe.get("depth") or {}).get("footprint") if pe else None,
    }

    hist = st.session_state.get("operator_history", [])
    hist.append(record)

    cutoff = ts - timedelta(minutes=history_window_minutes)
    hist = [r for r in hist if r.get("_ts") and r["_ts"] >= cutoff]
    st.session_state["operator_history"] = hist

def render_history(history_window_minutes: int):
    st.subheader(f"⏱ Last {history_window_minutes} minutes")
    hist = st.session_state.get("operator_history", [])
    if not hist:
        st.info("History will populate as the app refreshes.")
        return

    dfh = pd.DataFrame(hist).drop(columns=["_ts"], errors="ignore")

    for c in ["NIFTY %", "CE %", "PE %"]:
        if c in dfh.columns:
            dfh[c] = dfh[c].apply(lambda x: "-" if x is None or pd.isna(x) else f"{float(x):.2f}%")

    st.dataframe(dfh, use_container_width=True, hide_index=True)
    st.caption("Session-only memory; resets if Streamlit restarts.")

# =========================
# UI HELPERS
# =========================
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
        return f"{int(float(x))}"
    except Exception:
        return "-"

def render_header():
    st.title(f"{APP_TITLE} 🧲")
    st.caption(
        "Detects divergence between NIFTY and heavyweights + deep ITM CE/PE footprints. "
        "Token refresh is integrated in the sidebar."
    )

def render_snapshot(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No data returned.")
        return None

    nr = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if nr.empty:
        st.warning("NIFTY row missing.")
        return None

    r = nr.iloc[0]
    ts: datetime = r["Timestamp"]
    ts_str = ts.astimezone(TZ).strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)

    st.subheader("📈 NIFTY Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY LTP", _fmt_price(r["LTP"]))
    c2.metric("NIFTY %", _fmt_pct(r["% Change"]))
    c3.write(f"Timestamp (IST): {ts_str}")

    return float(r["LTP"]) if r["LTP"] is not None and not pd.isna(r["LTP"]) else None

def render_suppression(stats: dict | None, beep_enabled: bool):
    st.subheader("🧲 NIFTY vs Heavyweights")

    if not stats:
        st.info("Not enough clean data yet.")
        return

    supp = stats["supp_label"]
    infl = stats["infl_label"]

    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY %", _fmt_pct(stats["nifty_change"]))
    c2.metric("Avg Heavyweights %", _fmt_pct(stats["avg_heavy_change"]))
    c3.metric("Heavy - NIFTY", _fmt_pct(stats["divergence"]))

    is_high = (supp == "HIGH") or (infl == "HIGH")
    _beep_on_high_transition(is_high=is_high, enabled=beep_enabled)

    if supp == "HIGH":
        st.error("Suppression: HIGH")
    elif supp == "MILD":
        st.warning("Suppression: MILD")
    else:
        st.info("Suppression: NORMAL")

    if infl == "HIGH":
        st.error("Inflation: HIGH")
    elif infl == "MILD":
        st.warning("Inflation: MILD")
    else:
        st.info("Inflation: NORMAL")

def render_option_card(title: str, opt: dict | None, nifty_change: float | None, side: str):
    st.subheader(title)

    if not opt:
        st.info("Not available.")
        return "neutral", "NONE"

    depth = opt.get("depth") or {}
    strike = int(opt["strike"])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Symbol", opt["tradingsymbol"])
        st.metric("Strike", str(strike))
    with c2:
        st.metric("Expiry", str(opt["expiry"]))

    c3, c4 = st.columns(2)
    c3.metric("LTP", _fmt_price(opt.get("ltp")))
    c4.metric("LTQ", _fmt_int(opt.get("last_quantity")))

    vol15 = compute_recent_volume_15s(opt["instrument"], opt.get("volume_total"))

    c5, c6, c7 = st.columns(3)
    c5.metric("%", _fmt_pct(opt.get("pct_change")))
    c6.metric("NIFTY %", _fmt_pct(nifty_change))
    c7.metric("Est Vol (15s)", _fmt_int(vol15) if vol15 is not None else "-")

    if side == "CE":
        div = classify_ce_divergence(opt.get("pct_change"), nifty_change)
    else:
        div = classify_pe_divergence(opt.get("pct_change"), nifty_change)

    if div == "strong":
        st.error(f"Divergence: STRONG (Strike {strike})")
    elif div == "mild":
        st.warning(f"Divergence: MILD (Strike {strike})")
    else:
        st.info(f"Divergence: NEUTRAL (Strike {strike})")

    st.markdown("**Order Book (Top 5 depth)**")
    b1, b2, b3 = st.columns(3)
    b1.metric("Bid Qty", _fmt_int(depth.get("bid_qty")))
    b2.metric("Ask Qty", _fmt_int(depth.get("ask_qty")))
    ratio = depth.get("bid_ask_ratio")
    b3.metric("Bid/Ask", "-" if ratio in (0.0, float("inf")) else f"{ratio:.2f}")

    p1, p2 = st.columns(2)
    p1.metric("Top Bid", _fmt_price(depth.get("top_bid")))
    p2.metric("Top Ask", _fmt_price(depth.get("top_ask")))

    fp = depth.get("footprint", "NONE")
    if fp == "STRONG":
        st.error("Footprint: STRONG")
    elif fp == "MILD":
        st.warning("Footprint: MILD")
    else:
        st.info("Footprint: NONE")

    return div, fp

def render_heavyweights_table(df: pd.DataFrame):
    st.subheader("🏋️ Heavyweights")
    if df is None or df.empty:
        st.info("No rows")
        return

    t = df.copy()
    t["LTP"] = t["LTP"].map(_fmt_price)
    t["Prev Close"] = t["Prev Close"].map(_fmt_price)
    t["% Change"] = t["% Change"].map(_fmt_pct)

    st.dataframe(t[["Symbol", "LTP", "Prev Close", "% Change"]], use_container_width=True, hide_index=True)

# =========================
# BURST MODE TRIGGER
# =========================
def detect_strong_signal(stats: dict | None, ce_div: str, ce_fp: str, pe_div: str, pe_fp: str) -> bool:
    if stats and (stats.get("supp_label") == "HIGH" or stats.get("infl_label") == "HIGH"):
        return True
    if ce_div == "strong" and ce_fp in ("MILD", "STRONG"):
        return True
    if pe_div == "strong" and pe_fp in ("MILD", "STRONG"):
        return True
    return False

# =========================
# MAIN
# =========================
def main():
    render_header()

    api_key, api_secret, access_token = _auth_sidebar()

    st.sidebar.header("⚙️ Settings")
    beep_enabled = st.sidebar.toggle("Audio alert on HIGH", value=True)
    burst_enabled = st.sidebar.toggle("Burst mode", value=True)

    base_refresh_seconds = st.sidebar.slider(
        "Base refresh (seconds)",
        min_value=5,
        max_value=60,
        value=DEFAULT_BASE_REFRESH_SECONDS,
        step=5,
    )

    history_window = st.sidebar.slider(
        "History window (minutes)",
        min_value=1,
        max_value=20,
        value=DEFAULT_HISTORY_WINDOW_MINUTES,
        step=1,
    )

    heavyweights = st.sidebar.multiselect(
        "Heavyweights universe",
        options=sorted(set(DEFAULT_HEAVYWEIGHTS)),
        default=DEFAULT_HEAVYWEIGHTS,
        help="Keep it tight; every extra symbol is extra load.",
    )

    if not access_token:
        st.warning("No access_token available. Refresh token in the sidebar to enable live data.")
        st.stop()

    kite = get_kite_client(api_key, access_token)

    try:
        nifty_opt_df = get_nifty_option_instruments(api_key, access_token)
    except Exception as e:
        st.error(f"Failed to fetch instruments list: {e}")
        st.stop()

    try:
        df = fetch_ltp_snapshot(kite, heavyweights)
    except Exception as e:
        st.error(f"Kite data fetch failed: {e}")
        st.info("If this looks like an auth/token error, refresh token from the sidebar.")
        st.stop()

    nifty_spot = render_snapshot(df)

    stats = compute_suppression_stats(df)
    render_suppression(stats, beep_enabled=beep_enabled)

    st.subheader("🎯 Options Operator Footprint — Deep ITM CE & PE")

    nifty_change = None
    nr = df[df["Symbol"] == NIFTY_INDEX_SYMBOL]
    if not nr.empty:
        nifty_change = nr.iloc[0]["% Change"]

    ce_info = None
    pe_info = None
    if nifty_spot is not None and not nifty_opt_df.empty:
        try:
            ce_info, pe_info = quote_deep_itm_pair(kite, nifty_opt_df, nifty_spot)
        except Exception as e:
            st.warning(f"Option quote failed: {e}")

    col_ce, col_pe = st.columns(2)
    with col_ce:
        ce_div, ce_fp = render_option_card("🎯 ≥100pt ITM CE — Dip Buying", ce_info, nifty_change, side="CE")
    with col_pe:
        pe_div, pe_fp = render_option_card("🩸 ≥100pt ITM PE — Ramp & Dump", pe_info, nifty_change, side="PE")

    render_heavyweights_table(df)

    update_history(history_window, df, stats, ce_info, pe_info)
    render_history(history_window)

    strong = detect_strong_signal(stats, ce_div, ce_fp, pe_div, pe_fp)

    if burst_enabled and strong:
        effective = BURST_REFRESH_SECONDS
        st.sidebar.warning(f"🔥 Burst active → ~{effective}s")
    else:
        effective = base_refresh_seconds
        st.sidebar.info(f"Refresh: ~{effective}s")

    st_autorefresh(interval=int(effective * 1000), key="auto_refresh")

if __name__ == "__main__":
    main()
