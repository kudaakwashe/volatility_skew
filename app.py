# app.py — Skew Timeseries from local Dolt SQL server (table: option_chain)
# DISCLAIMER: For educational/research use only. Not investment advice.

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
import yfinance as yf

# ───────────────────────── Streamlit UI ─────────────────────────
st.set_page_config(page_title="Skew Timeseries — Local Dolt", layout="wide")
st.title("Skew Timeseries (25Δ Call/Put vs ATM) — Local Dolt")
st.caption("Call skew = (ATM IV − 25Δ Call IV) / ATM IV • Put skew = (ATM IV − 25Δ Put IV) / ATM IV • Nearest 25Δ via stored `delta`.")

with st.sidebar:
    st.subheader("Local Dolt connection")
    st.write("Configure `.streamlit/secrets.toml` (example below).")
    dolt_db = st.text_input("Database", value=st.secrets.get("dolt", {}).get("database", "options"))
    dolt_branch = st.text_input("Branch (optional)", value="")   # e.g., master
    dolt_commit = st.text_input("Commit hash (optional)", value="")  # read-only at commit

    st.subheader("Query params")
    ticker = st.text_input("Ticker (act_symbol)", value="A").strip().upper()
    target_dte = st.number_input("Target DTE (days)", 7, 90, 30, 1)
    months = st.slider("Lookback (months)", 1, 24, 6)
    table = st.text_input("Table", value="option_chain")
    dates_are_text = st.checkbox("`date` is TEXT like YYYY/MM/DD", value=False)

asof = datetime.now(timezone.utc)
start_dt = asof - timedelta(days=30*months)
end_dt   = asof

# ───────────────────────── DB connection ─────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine_from_secrets(db_name: str):
    """Expect secrets:
    [dolt]
    host="127.0.0.1"
    port=3306
    database="options"
    user="root"
    password=""
    """
    s = st.secrets.get("dolt", {})
    host = s.get("host", "127.0.0.1")
    port = int(s.get("port", 3306))
    user = s.get("user", "root")
    password = s.get("password", "")
    uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(uri, pool_pre_ping=True, pool_recycle=3600)

engine = get_engine_from_secrets(dolt_db)

def select_branch_or_commit(conn, branch: str, commit: str):
    """Use Dolt session procs/vars to move the read view."""
    try:
        if branch.strip():
            conn.execute(text("CALL DOLT_CHECKOUT(:b)"), {"b": branch.strip()})
        if commit.strip():
            conn.execute(text("SET @@dolt_read_from_commit = :h"), {"h": commit.strip()})
    except Exception as e:
        st.warning(f"Branch/commit selection failed; reading current HEAD. Detail: {e}")

# ───────────────────────── Data fetch ─────────────────────────
@st.cache_data(ttl=600, show_spinner=True)
def fetch_option_chain(symbol: str, start_dt: datetime, end_dt: datetime,
                       table: str, branch: str, commit: str, dates_are_text: bool) -> pd.DataFrame:
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso   = end_dt.strftime("%Y-%m-%d")

    if dates_are_text:
        date_filter = "STR_TO_DATE(`date`, '%Y/%m/%d') >= :s AND STR_TO_DATE(`date`, '%Y/%m/%d') <= :e"
    else:
        date_filter = "`date` >= :s AND `date` <= :e"

    q = text(f"""
        SELECT `date`,`act_symbol`,`expiration`,`strike`,`call_put`,`vol`,`delta`
        FROM {table}
        WHERE act_symbol = :tic
          AND {date_filter}
    """)

    with engine.begin() as conn:
        select_branch_or_commit(conn, branch, commit)
        df = pd.read_sql(q, conn, params={"tic": symbol, "s": start_iso, "e": end_iso})

    for col in ["date", "expiration"]:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.isna().all():
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True, format="%Y/%m/%d")
        df[col] = parsed

    df["call_put"] = df["call_put"].astype(str).str.capitalize().str[:3]
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    for c in ["vol","delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date","expiration","strike"]).drop_duplicates()
    df = df.sort_values(["date","expiration","strike"]).reset_index(drop=True)
    return df

# ───────────────────────── Spot fill ─────────────────────────
def fill_spot_from_yahoo(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    px = yf.Ticker(symbol).history(period="2y")["Close"].rename("spot").to_frame()
    px.index = pd.to_datetime(px.index.date)
    out = df.copy()
    out["asof_date"] = pd.to_datetime(out["date"].dt.date)
    out = out.merge(px, left_on="asof_date", right_index=True, how="left")
    return out

# ───────────────────────── Skew calc ─────────────────────────
def compute_skew_for_day(day_df: pd.DataFrame, snap_dt: pd.Timestamp, target_dte: int):
    snap_dt = pd.Timestamp(snap_dt).tz_localize(None).normalize()
    day = day_df.copy()
    day["expiration"] = pd.to_datetime(day["expiration"]).dt.tz_localize(None).dt.normalize()
    day["date"] = pd.to_datetime(day["date"]).dt.tz_localize(None).dt.normalize()
    day["dte"] = (day["expiration"] - snap_dt).dt.days

    cand = day.loc[day["dte"] >= 1, ["expiration","dte"]].drop_duplicates()
    if cand.empty:
        return None
    pick_exp = cand.iloc[(cand["dte"] - target_dte).abs().argsort()[:1]]["expiration"].values[0]

    S = day["spot"].dropna().mean()
    if not (np.isfinite(S) and S > 0):
        return None

    calls = day[(day["expiration"]==pick_exp) & (day["call_put"]=="Call")].copy()
    puts  = day[(day["expiration"]==pick_exp) & (day["call_put"]=="Put") ].copy()
    if calls.empty and puts.empty:
        return None

    atm_strike_candidates = pd.concat([calls[["strike"]], puts[["strike"]]], ignore_index=True)
    if atm_strike_candidates.empty:
        return None
    atm_strike = atm_strike_candidates.iloc[(atm_strike_candidates["strike"] - S).abs().argsort()[:1]]["strike"].values[0]

    def iv_at_strike(df, K):
        if df is None or df.empty:
            return np.nan
        sub = df.iloc[(df["strike"] - K).abs().argsort()[:1]]
        return sub["vol"].values[0] if len(sub) else np.nan

    atm_call_iv = iv_at_strike(calls, atm_strike)
    atm_put_iv  = iv_at_strike(puts,  atm_strike)
    iv_atm = np.nanmean([atm_call_iv, atm_put_iv])

    call25_iv = np.nan
    if not calls.empty and "delta" in calls:
        c = calls.dropna(subset=["delta"])
        if not c.empty:
            call25_iv = c.iloc[(c["delta"] - 0.25).abs().argsort()[:1]]["vol"].values[0]
    put25_iv = np.nan
    if not puts.empty and "delta" in puts:
        p = puts.dropna(subset=["delta"])
        if not p.empty:
            put25_iv = p.iloc[(p["delta"].abs() - 0.25).abs().argsort()[:1]]["vol"].values[0]

    call_skew = (iv_atm - call25_iv) / iv_atm if (np.isfinite(iv_atm) and iv_atm > 0 and np.isfinite(call25_iv)) else np.nan
    put_skew  = (iv_atm - put25_iv)  / iv_atm if (np.isfinite(iv_atm) and iv_atm > 0 and np.isfinite(put25_iv))  else np.nan
    rr = call_skew - put_skew if np.isfinite(call_skew) and np.isfinite(put_skew) else np.nan

    return {
        "asof": pd.to_datetime(snap_dt.date()),
        "expiration": pd.to_datetime(pd.Timestamp(pick_exp).date()),
        "dte": int((pd.Timestamp(pick_exp).date() - snap_dt.date()).days),
        "spot": S,
        "atm_iv": iv_atm,
        "call25_iv": call25_iv,
        "put25_iv":  put25_iv,
        "call_skew": call_skew,
        "put_skew":  put_skew,
        "rr": rr
    }

def build_timeseries(df: pd.DataFrame, symbol: str, target_dte: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = fill_spot_from_yahoo(df, symbol)
    out = []
    for snap_dt, day_df in df.groupby(df["date"].dt.normalize()):
        res = compute_skew_for_day(day_df, snap_dt, target_dte)
        if res:
            out.append(res)
    return pd.DataFrame(out).sort_values("asof")

def plot_skew_timeseries(ts: pd.DataFrame, mode="Call"):
    if ts is None or ts.empty:
        return go.Figure()
    col = {"Call": "call_skew", "Put": "put_skew", "RR": "rr"}[mode]
    d = ts[["asof", col]].dropna().sort_values("asof")
    mu, sig = d[col].mean(), d[col].std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["asof"], y=d[col], mode="lines+markers", name=f"{mode} Skew",
        hovertemplate="%{x|%b %d, %Y}<br>"+f"{mode} Skew: %{y:.2%}"
    ))
    for name, val, dash in [("Mean", mu, "dash"), ("+1σ", mu+sig, "dot"), ("-1σ", mu-sig, "dot")]:
        fig.add_hline(y=float(val), line_dash=dash, annotation_text=name, annotation_position="right")
    fig.update_layout(template="plotly_dark", height=440, margin=dict(l=30,r=20,t=50,b=30),
                      title=f"Skew Timeseries — {mode}", xaxis_title="Date",
                      yaxis_title=f"{mode} Skew (fraction)", hovermode="x unified")
    return fig

# ───────────────────────── Run ─────────────────────────
with st.spinner("Querying local Dolt and computing skews…"):
    rows = fetch_option_chain(ticker, start_dt, end_dt, table, dolt_branch, dolt_commit, dates_are_text)
    ts = build_timeseries(rows, ticker, target_dte)

left, right = st.columns([1,1])
with left:
    st.subheader(f"{ticker} • {months}m • DTE≈{target_dte}d")
    if ts.empty:
        st.info("No timeseries built. Check DB/branch/commit, table/columns, and date format toggle.")
    else:
        st.plotly_chart(plot_skew_timeseries(ts, "Call"), use_container_width=True)
with right:
    if not ts.empty:
        st.plotly_chart(plot_skew_timeseries(ts, "Put"), use_container_width=True)

st.markdown("### Summary")
if not ts.empty:
    st.dataframe(
        ts.sort_values("asof", ascending=False)[
            ["asof","expiration","dte","spot","atm_iv","call25_iv","put25_iv","call_skew","put_skew","rr"]
        ].style.format({
            "spot":"{:.2f}",
            "atm_iv":"{:.2%}","call25_iv":"{:.2%}","put25_iv":"{:.2%}",
            "call_skew":"{:.2%}","put_skew":"{:.2%}","rr":"{:.2%}"
        }),
        use_container_width=True, height=360
    )
