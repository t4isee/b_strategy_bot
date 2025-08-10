cat > bot.py << 'EOF'
import os, json, time, math, requests, datetime as dt
import pandas as pd, numpy as np, yfinance as yf, pytz, yaml
from slack_sdk.webhook import WebhookClient

# ---- ENV / CONFIG ----
JST = pytz.timezone("Asia/Tokyo")
NY  = pytz.timezone("America/New_York")
PAIR = os.getenv("PAIR", "USDJPY=X")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
CONFIG_PATH = os.getenv("CONFIG_PATH", "strategy_B_dd_tuned.yaml")
STATE_PATH = os.getenv("STATE_PATH", ".state.json")

# ---- Slack ----
def slack_post(text: str):
    if not WEBHOOK_URL:
        print("[WARN] SLACK_WEBHOOK_URL 未設定。メッセージ:", text)
        return
    WebhookClient(WEBHOOK_URL).send(text=text)

# ---- Utils ----
def now_jst():
    return dt.datetime.now(tz=JST)

def load_state():
    try:
        with open(STATE_PATH, "r") as f: return json.load(f)
    except Exception:
        return {}

def save_state(s):
    with open(STATE_PATH, "w") as f: json.dump(s, f)

def in_time_windows_jst(ts: pd.Timestamp, ranges):
    """tsはJSTのTimestamp。['13:00-23:59','00:00-02:00'] みたいな配列を評価。"""
    m = ts.hour*60 + ts.minute
    for r in ranges:
        start,end = r.split("-")
        sh,sm = map(int, start.split(":")); eh,em = map(int, end.split(":"))
        smin = sh*60+sm; emin = eh*60+em
        if emin >= smin:
            if smin <= m <= emin: return True
        else:
            if (m >= smin) or (m <= emin): return True
    return False

def wilder_atr(tr, period=14):
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_vwap_session_TR(df):
    """NY 17:00 リセットの擬似VWAP（重み=TR）"""
    idx_ny = df.index.tz_convert(NY)
    session_key = (idx_ny - pd.Timedelta(hours=17)).floor("1D")  # 17時で日替わり
    vol_proxy = df["TR"].clip(lower=1e-9)  # TRを重み
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["_pv"] = (tp * vol_proxy).groupby(session_key).cumsum()
    df["_v"]  = (vol_proxy).groupby(session_key).cumsum()
    vwap = df["_pv"] / df["_v"]
    df.drop(columns=["_pv","_v"], inplace=True)
    return vwap

def read_yaml_or_default(path):
    default = {
        "FILTERS":{
            "time_windows":["13:00-23:59","00:00-02:00"],
            "atr_range_jpy":[0.04, 0.12],
            "atr_top_skip_pct":0.90,
            "vwap_deviation_atr":2.0,
            "h1_alignment":True,
            "volume_confirm_ratio":1.2
        },
        "ENTRY":{
            "breakout_donchian_h1_buffer_atr":0.25,
        },
        "RISK_MANAGEMENT":{
            "sl_atr_mult":1.1,
            "tp1_atr_mult":1.3,
            "shock_bar_filter_TR_over_ATR":1.8
        }
    }
    try:
        with open(path, "r") as f:
            y = yaml.safe_load(f)
            # 既存キーにマージ（無ければdefault）
            for k in default:
                if k not in y: y[k]=default[k]
                else:
                    for kk in default[k]:
                        if kk not in y[k]: y[k][kk]=default[k][kk]
            return y
    except Exception:
        return default

# ---- Indicators ----
def compute_indicators(df, cfg):
    # インデックスはすでにJST tz-aware
    # TR / ATR
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"]-df["Low"]).abs(),
        (df["High"]-prev_close).abs(),
        (df["Low"]-prev_close).abs()
    ], axis=1).max(axis=1)
    df["TR"] = tr
    df["ATR14"] = wilder_atr(tr, 14)

    # EMA9 / EMA20
    df["EMA9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI14
    delta = df["Close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss+1e-12)
    df["RSI14"] = 100 - (100/(1+rs))

    # Donchian 20 (直近20本の高安) — 先読み回避のため1本シフト
    df["don_high"] = df["High"].rolling(20).max().shift(1)
    df["don_low"]  = df["Low"].rolling(20).min().shift(1)

    # VWAP (TR重み、NY17:00リセット)
    df["VWAP"] = compute_vwap_session_TR(df)

    # H1整合
    if cfg["FILTERS"]["h1_alignment"]:
        h1 = df["Close"].resample("1H").last().dropna()
        h1_ema50  = h1.ewm(span=50, adjust=False).mean()
        h1_ema200 = h1.ewm(span=200, adjust=False).mean()
        h1_long  = (h1_ema50 > h1_ema200)
        h1_short = (h1_ema50 < h1_ema200)
        df["H1_long"]  = h1_long.reindex(df.index, method="ffill")
        df["H1_short"] = h1_short.reindex(df.index, method="ffill")
    else:
        df["H1_long"]  = True
        df["H1_short"] = True

    # Volume確認（yfinanceはFX volumeが0/NaN多い→TRで代替）
    vol3 = df["TR"].rolling(3).mean()
    volMed20 = df["TR"].rolling(20).median()
    df["VOL_OK"] = (vol3 >= volMed20 * cfg["FILTERS"]["volume_confirm_ratio"])

    # ATR上位%スキップ用の分位
    look = 24*4*30  # 30日相当
    q = df["ATR14"].rolling(look, min_periods=24*4*7).quantile(cfg["FILTERS"]["atr_top_skip_pct"])
    df["ATR_TOP_OK"] = (df["ATR14"] <= q.fillna(np.inf))

    return df

# ---- Signal ----
def last_signal(df, cfg):
    # 最新バー（確定）のみ判断
    last = df.iloc[-1]
    jst_ts = last.name.tz_convert(JST)

    # 時間帯フィルタ
    if not in_time_windows_jst(jst_ts, cfg["FILTERS"]["time_windows"]):
        return 0, last, "time_filter_ng"

    # ATRレンジ
    amin, amax = cfg["FILTERS"]["atr_range_jpy"]
    if not (amin <= last["ATR14"] <= amax):
        return 0, last, "atr_range_ng"

    # ATR上位%スキップ
    if not bool(last["ATR_TOP_OK"]):
        return 0, last, "atr_top_skip"

    # VWAP乖離
    vdev = abs(last["Close"] - last["VWAP"])
    if vdev > cfg["FILTERS"]["vwap_deviation_atr"] * last["ATR14"]:
        return 0, last, "vwap_dev_ng"

    # ショックバー除外
    shock = last["TR"] / (last["ATR14"]+1e-12)
    if shock >= cfg["RISK_MANAGEMENT"]["shock_bar_filter_TR_over_ATR"]:
        return 0, last, "shock_ng"

    # トレンド要件
    trend_long  = (last["EMA9"] > last["EMA20"]) and (last["Close"] > last["VWAP"]) and (last["MACD"] > 0) and (last["RSI14"] >= 50) and bool(last["H1_long"])
    trend_short = (last["EMA9"] < last["EMA20"]) and (last["Close"] < last["VWAP"]) and (last["MACD"] < 0) and (last["RSI14"] <= 50) and bool(last["H1_short"])

    # ブレイク（Donchian + 0.25ATR）
    up_break = last["Close"] > (last["don_high"] + cfg["ENTRY"]["breakout_donchian_h1_buffer_atr"]*last["ATR14"])
    dn_break = last["Close"] < (last["don_low"]  - cfg["ENTRY"]["breakout_donchian_h1_buffer_atr"]*last["ATR14"])

    if trend_long and up_break:
        return 1, last, "long_ok"
    if trend_short and dn_break:
        return -1, last, "short_ok"
    return 0, last, "no_setup"

# ---- Main tick ----
def run_once():
    cfg = read_yaml_or_default(CONFIG_PATH)

    # 直近データ取得（15分足。yfinanceは直近60日程度まで）
    # periodは余裕を持って‘90d’。失敗時は再試行。
    df = yf.download(PAIR, interval="15m", period="90d", progress=False)
    if df.empty:
        print("[WARN] yfinanceデータ空。リトライ待ち。")
        return

    # 整形
    df = df.rename(columns=str).rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
    if df.index.tz is None:
        # yfinanceはUTC
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert(JST)  # まずJSTに
    # そのままJSTのまま計算（H1整合は内部でNYに変換して扱う）
    df = compute_indicators(df, cfg).dropna().copy()

    # シグナル判定（最新バー）
    sig, bar, reason = last_signal(df, cfg)

    # 15分足が新しく確定したかどうかの重複防止
    state = load_state()
    last_ts_str = state.get("last_ts")
    this_ts_str = bar.name.isoformat()
    if last_ts_str == this_ts_str:
        # すでに処理済み
        return

    if sig == 0:
        # 更新だけして終わり
        state["last_ts"] = this_ts_str
        save_state(state)
        print(f"[{now_jst().strftime('%Y-%m-%d %H:%M')}] no signal ({reason})")
        return

    # 推奨エントリー：次バー始値想定 → ここでは参考値として現Closeを使ってレベルだけ出す
    price = float(bar["Close"])
    atr   = float(bar["ATR14"])
    slmul = cfg["RISK_MANAGEMENT"]["sl_atr_mult"]
    tp1mul= cfg["RISK_MANAGEMENT"]["tp1_atr_mult"]

    if sig > 0:
        side = "BUY"
        sl = price - slmul*atr
        tp1= price + tp1mul*atr
        emoji = "🟢"
    else:
        side = "SELL"
        sl = price + slmul*atr
        tp1= price - tp1mul*atr
        emoji = "🔴"

    # pips表記（USDJPYは0.01=1pip）
    pips = lambda x: round(x/0.01, 1)

    msg = (
        f"{emoji} *B strategy (DD tuned)* — {PAIR} / 15m\n"
        f"*Signal*: {side}\n"
        f"*Time (JST)*: {bar.name.tz_convert(JST).strftime('%Y-%m-%d %H:%M')}\n"
        f"*Price*: {price:.3f}\n"
        f"*ATR14*: {atr:.3f} （{pips(atr)} pips）\n"
        f"*SL*: {sl:.3f}  (≈ {pips(abs(price-sl))} pips)\n"
        f"*TP1*: {tp1:.3f} (≈ {pips(abs(tp1-price))} pips)\n"
        f"*Notes*: reason={reason}, VWAP_dev≤{cfg['FILTERS']['vwap_deviation_atr']}×ATR, shock<{cfg['RISK_MANAGEMENT']['shock_bar_filter_TR_over_ATR']}\n"
        "_Not financial advice_"
    )
    slack_post(msg)

    state["last_ts"] = this_ts_str
    save_state(state)
    print(f"[{now_jst().strftime('%Y-%m-%d %H:%M')}] sent: {side} {price:.3f}")

if __name__ == "__main__":
    # 毎分起動（内部で重複ガード&バー確定でのみ通知）
    while True:
        try:
            # 0,15,30,45分のタイミングに近いときだけ重い処理
            m = now_jst().minute
            if m % 15 == 0:
                run_once()
            else:
                # 軽くする
                pass
        except Exception as e:
            slack_post(f"⚠️ B-bot error: {e}")
            print("Error:", e)
        time.sleep(30)
EOF
