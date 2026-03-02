import sys
import signal as sig
import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv
import pandas_ta as ta

# =========================================================
# 0) 로깅/환경변수
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trade.log", encoding="utf-8"), logging.StreamHandler()],
)
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# =========================================================
# 1) 기본 설정
# =========================================================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

LEVERAGE = 5

# 자금/리스크 관리
BET_RATE = 0.05            # 최대 노출 캡: equity * BET_RATE * LEVERAGE
RISK_PER_TRADE = 0.005     # 1회 트레이드 리스크(손실 허용) = equity * RISK_PER_TRADE
MAX_OPEN_POSITIONS = 1     # 동시 포지션 제한

# 전략 파라미터 (15m 기준 중심선/엔벨로프)
WINDOW_SIZE = 50           # 15분봉 SMA50
ENVELOPE_PERCENT = 0.01    # 중심선 기준 ±1%

# 필터 스위치
ENABLE_TREND_FILTER = False
ENABLE_VOLATILITY_FILTER = False
ENABLE_VOLUME_FILTER = False

# 필터 기준
TREND_TF = "1h"
TREND_SMA = 50

VOL_TF = "15m"
VOL_LOOKBACK = 14
VOL_MIN_ATR_PCT = 0.10

VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.5

# 실행 루프 주기
SYMBOL_SLEEP_SEC = 1.5
LOOP_SLEEP_SEC = 30
ERROR_SLEEP_SEC = 10

# =========================================================
# 1-1) Market Regime (국면) 설정 (하이브리드 최적화)
# =========================================================
REGIME_TF = "1h"

# 1. DI 스프레드 변수
REGIME_ADX_LEN = 14
REGIME_DI_SPREAD_ON = 15   # 두 선의 격차가 15 포인트 이상 벌어지면 트렌드 진입
REGIME_DI_SPREAD_OFF = 10  # 두 선의 격차가 10 포인트 이하로 좁혀지면 횡보장 복귀

# 2. SMA 기울기 변수 (안전벨트)
REGIME_SMA_LEN = 20
REGIME_SLOPE_LOOKBACK = 3
REGIME_SLOPE_PCT_TH = 0.01 # 기울기가 0.10% 이상 틀어져야 진짜 방향으로 인정

# 국면 캐시 
REGIME_CACHE_TTL_SEC = 300

# =========================================================
# 2) 거래소 초기화
# =========================================================
exchange = ccxt.binance(
    {
        "apiKey": API_KEY,
        "secret": SECRET_KEY,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)
exchange.load_markets()

# =========================================================
# 3) 유틸: 안전 호출/알림/정밀도/최소주문
# =========================================================
def send_n8n(msg: str) -> None:
    if not N8N_WEBHOOK_URL:
        return
    try:
        r = requests.post(N8N_WEBHOOK_URL, json={"message": msg}, timeout=5)
        if r.status_code >= 400:
            logging.warning(f"N8N 응답 오류({r.status_code}): {msg}")
    except Exception as e:
        logging.warning(f"N8N 전송 실패: {e} | msg={msg}")


def safe_call(fn, *args, retries: int = 3, base_sleep: float = 0.6, **kwargs):
    """거래소/네트워크 오류에 대비한 재시도 래퍼."""
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            sleep = base_sleep * (2 ** i)
            logging.warning(f"네트워크 오류 재시도({i+1}/{retries}) {fn.__name__}: {e} -> {sleep:.1f}s")
            time.sleep(sleep)
        except ccxt.ExchangeError as e:
            sleep = base_sleep * (2 ** i)
            logging.warning(f"거래소 오류 재시도({i+1}/{retries}) {fn.__name__}: {e} -> {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"safe_call 실패: {fn.__name__} ({retries} retries)")


def get_futures_usdt_equity() -> float:
    bal = safe_call(exchange.fetch_balance, params={"type": "future"})
    if isinstance(bal, dict):
        usdt = bal.get("USDT")
        if isinstance(usdt, dict):
            total = usdt.get("total")
            free = usdt.get("free")
            if total is not None:
                return float(total)
            if free is not None:
                return float(free)

        info = bal.get("info")
        if isinstance(info, dict):
            for k in ["totalWalletBalance", "availableBalance", "totalMarginBalance"]:
                v = info.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except:
                        pass
    return 0.0


def get_market_min_notional(symbol: str) -> float:
    m = exchange.market(symbol)
    try:
        v = m.get("limits", {}).get("cost", {}).get("min", None)
        if v is not None:
            return float(v)
    except:
        pass

    info = m.get("info", {})
    if isinstance(info, dict):
        for k in ["minNotional", "notionalMin", "min_notional", "minTradeAmount"]:
            if k in info:
                try:
                    return float(info[k])
                except:
                    pass
    return 5.0


def amount_precision(symbol: str, amount: float) -> float:
    return float(exchange.amount_to_precision(symbol, amount))


def price_precision(symbol: str, price: float) -> float:
    return float(exchange.price_to_precision(symbol, price))

# =========================================================
# 4) 지표/필터
# =========================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["h"], df["l"], df["c"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


@dataclass
class TrendCacheItem:
    ts: float
    mode: str  # 'LONG_ONLY' | 'SHORT_ONLY' | 'BOTH'


trend_cache: Dict[str, TrendCacheItem] = {}
TREND_CACHE_TTL_SEC = 300


def check_trend_filter(symbol: str) -> str:
    if not ENABLE_TREND_FILTER:
        return "BOTH"

    now = time.time()
    cached = trend_cache.get(symbol)
    if cached and (now - cached.ts) < TREND_CACHE_TTL_SEC:
        return cached.mode

    ohlcv = safe_call(exchange.fetch_ohlcv, symbol, timeframe=TREND_TF, limit=TREND_SMA + 10)
    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    if len(df) < TREND_SMA + 2:
        mode = "BOTH"
    else:
        sma = df["c"].rolling(TREND_SMA).mean().iloc[-2]
        close = df["c"].iloc[-2]
        mode = "LONG_ONLY" if close > sma else "SHORT_ONLY"

    trend_cache[symbol] = TrendCacheItem(ts=now, mode=mode)
    return mode


def check_volatility_filter(df_5m: pd.DataFrame) -> bool:
    if not ENABLE_VOLATILITY_FILTER:
        return True
    atr = compute_atr(df_5m, period=VOL_LOOKBACK)
    atr_last = atr.iloc[-2]
    price_last = df_5m["c"].iloc[-2]
    if pd.isna(atr_last) or price_last <= 0:
        return False
    return ((atr_last / price_last) * 100.0) >= VOL_MIN_ATR_PCT


def check_volume_filter(df_5m: pd.DataFrame) -> bool:
    if not ENABLE_VOLUME_FILTER:
        return True
    if len(df_5m) < VOLUME_LOOKBACK + 3:
        return False

    vols = df_5m["v"]
    baseline = vols.iloc[-(VOLUME_LOOKBACK + 2) : -2].mean()
    last_vol = vols.iloc[-2]

    if baseline <= 0:
        return False
    return last_vol >= baseline * VOLUME_SPIKE_MULT

# =========================================================
# 4-1) Market Regime (국면 판별)
# =========================================================
@dataclass
class RegimeCacheItem:
    ts: float
    regime: str  # 'TREND_UP' | 'TREND_DOWN' | 'RANGING'


regime_cache: Dict[str, RegimeCacheItem] = {}


def check_market_regime(symbol: str) -> str:
    now = time.time()
    cached = regime_cache.get(symbol)
    if cached and (now - cached.ts) < REGIME_CACHE_TTL_SEC:
        return cached.regime

    limit = 300
    ohlcv = safe_call(exchange.fetch_ohlcv, symbol, timeframe=REGIME_TF, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

    if len(df) < max(REGIME_SMA_LEN, REGIME_ADX_LEN) + REGIME_SLOPE_LOOKBACK + 5:
        return "RANGING"

    # ⭐ pandas_ta의 치명적 버그(5000+ 표기)를 피해, 순수 Pandas로 DMI를 계산합니다. (0~100 스케일 완벽 보장)
    # 1. TR (True Range)
    df['h_l'] = df['h'] - df['l']
    df['h_pc'] = (df['h'] - df['c'].shift(1)).abs()
    df['l_pc'] = (df['l'] - df['c'].shift(1)).abs()
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

    # 2. 방향 이동 (+DM, -DM)
    df['up_move'] = df['h'] - df['h'].shift(1)
    df['down_move'] = df['l'].shift(1) - df['l']
    
    df['plus_dm'] = 0.0
    df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), 'plus_dm'] = df['up_move']
    
    df['minus_dm'] = 0.0
    df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), 'minus_dm'] = df['down_move']

    # 3. Wilder's Smoothing (RMA 방식)
    alpha = 1.0 / REGIME_ADX_LEN
    df['tr_rma'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_rma'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_rma'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()

    # 4. +DI, -DI 계산 (0~100 백분율)
    df['plus_di'] = 100.0 * (df['plus_dm_rma'] / (df['tr_rma'] + 1e-10))
    df['minus_di'] = 100.0 * (df['minus_dm_rma'] / (df['tr_rma'] + 1e-10))

    # 5. SMA 20 (방향성 확인용 안전벨트)
    sma = df["c"].rolling(REGIME_SMA_LEN).mean()

    # --------------- 신호 판별 로직 ---------------
    i = -2  # 확정봉 기준
    plus_di = float(df['plus_di'].iloc[i])
    minus_di = float(df['minus_di'].iloc[i])
    sma_now = float(sma.iloc[i])
    sma_past = float(sma.iloc[i - REGIME_SLOPE_LOOKBACK])

    if math.isnan(plus_di) or math.isnan(sma_now) or math.isnan(sma_past) or sma_past == 0:
        return "RANGING"

    # [1] DI 스프레드 (단기 모멘텀/힘)
    di_diff = plus_di - minus_di
    
    # [2] SMA 기울기 (거시적 방향성)
    slope_pct = (sma_now - sma_past) / sma_past * 100.0

    prev_regime = cached.regime if cached else "RANGING"

    # 히스테리시스 (이중 잠금장치: 스프레드와 기울기 방향이 완벽히 일치할 때만 진입!)
    if prev_regime == "RANGING":
        if di_diff >= REGIME_DI_SPREAD_ON and slope_pct >= REGIME_SLOPE_PCT_TH:
            regime = "TREND_UP"
        elif di_diff <= -REGIME_DI_SPREAD_ON and slope_pct <= -REGIME_SLOPE_PCT_TH:
            regime = "TREND_DOWN"
        else:
            regime = "RANGING"
            
    elif prev_regime == "TREND_UP":
        # 롱 포지션 중 매수세가 약해져 OFF 기준(10) 이하로 떨어지면 횡보장 전환
        if di_diff <= REGIME_DI_SPREAD_OFF:
            regime = "RANGING"
        else:
            regime = "TREND_UP"
            
    elif prev_regime == "TREND_DOWN":
        # 숏 포지션 중 매도세가 약해져 OFF 기준(-10) 이상으로 올라오면 횡보장 전환
        if di_diff >= -REGIME_DI_SPREAD_OFF:
            regime = "RANGING"
        else:
            regime = "TREND_DOWN"

    regime_cache[symbol] = RegimeCacheItem(ts=now, regime=regime)

    logging.info(
        f"[{symbol}] Regime={regime} | +DI={plus_di:.1f} | -DI={minus_di:.1f} | Spread={di_diff:.1f} | SMA_slope={slope_pct:.2f}%"
    )

    return regime

# =========================================================
# 5) 포지션/주문 관리
# =========================================================
def fetch_positions_map(symbols) -> Dict[str, Tuple[float, Optional[str]]]:
    positions = safe_call(exchange.fetch_positions, symbols)
    out = {s: (0.0, None) for s in symbols}

    for p in positions:
        psym = p.get("symbol", "")
        for s in symbols:
            if s in psym:
                contracts = None
                for k in ["contracts", "contractSize", "positionAmt"]:
                    if k in p and p[k] is not None:
                        try:
                            contracts = float(p[k])
                            break
                        except:
                            pass
                if contracts is None:
                    try:
                        contracts = float(p.get("contracts", 0))
                    except:
                        contracts = 0.0

                size_abs = abs(contracts)
                side = "LONG" if contracts > 0 else ("SHORT" if contracts < 0 else None)

                if size_abs > 0:
                    out[s] = (size_abs, side)
    return out


def set_symbol_leverage(symbol: str, lev: int) -> None:
    # 1. 마진 모드 명시적 설정 (격리: isolated, 교차: cross)
    try:
        # CCXT 최신 버전 표준 마진 설정 함수
        safe_call(exchange.set_margin_mode, 'isolated', symbol)
        logging.info(f"[{symbol}] 마진 모드 'ISOLATED(격리)' 설정 완료")
    except Exception as e:
        # 이미 Isolated 상태이거나, 포지션이 있는 상태에서 변경 시도 시 발생하는 에러는 패스
        if "No need to change" in str(e) or "margin type cannot be changed" in str(e).lower():
            pass
        else:
            logging.warning(f"[{symbol}] 마진 모드 설정 중 예외(무시됨): {e}")

    # 2. 레버리지 설정
    try:
        safe_call(exchange.set_leverage, lev, symbol)
        logging.info(f"[{symbol}] 레버리지 {lev}배 설정 완료")
    except Exception as e:
        logging.error(f"[{symbol}] 레버리지 설정 실패: {e}")


def calc_position_size(symbol: str, entry_price: float, stop_price: float, equity_usdt: float) -> float:
    if equity_usdt <= 0:
        return 0.0

    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0

    risk_usdt = equity_usdt * RISK_PER_TRADE
    raw_qty = risk_usdt / stop_dist

    max_notional = equity_usdt * BET_RATE * LEVERAGE
    qty_cap = max_notional / entry_price
    qty = min(raw_qty, qty_cap)

    min_notional = get_market_min_notional(symbol)
    min_qty_from_notional = min_notional / entry_price

    m = exchange.market(symbol)
    min_amount = m.get("limits", {}).get("amount", {}).get("min", 0) or 0

    qty = max(qty, float(min_qty_from_notional), float(min_amount))
    qty = amount_precision(symbol, qty)

    if qty * entry_price < min_notional:
        qty = amount_precision(symbol, qty + (min_qty_from_notional * 0.2))

    return float(qty)


def place_entry_and_brackets(symbol: str, signal: str, qty: float, tp: float, sl: float) -> None:
    side = "buy" if signal == "LONG" else "sell"
    exit_side = "sell" if side == "buy" else "buy"

    # 1) 진입 (시장가)
    try:
        safe_call(exchange.create_market_order, symbol, side, qty)
    except Exception as e:
        raise RuntimeError(f"시장가 진입 주문 실패: {e}")

    # 2) TP/SL 예약
    try:
        # ✅ TP는 지정가 (TAKE_PROFIT)
        # - price 자리에 '지정가로 내고 싶은 가격'을 넣어야 함
        # - stopPrice는 트리거 가격
        safe_call(
            exchange.create_order,
            symbol,
            "TAKE_PROFIT",
            exit_side,
            qty,
            tp,  # 지정가(실제 주문 가격)
            params={
                "stopPrice": tp,       # 트리거 가격
                "reduceOnly": True,
                "timeInForce": "GTC",
            },
        )

        # ✅ SL은 시장가 유지 (STOP_MARKET)
        safe_call(
            exchange.create_order,
            symbol,
            "STOP_MARKET",
            exit_side,
            qty,
            None,
            params={"stopPrice": sl, "reduceOnly": True},
        )

    except Exception as e:
        # TP/SL 예약 실패 -> 고아 포지션 방어(긴급 청산) 그대로 유지
        alert_msg = f"🚨 [{symbol}] TP/SL 예약 실패! 긴급 청산 시도. 에러: {e}"
        logging.error(alert_msg)
        send_n8n(alert_msg)

        try:
            safe_call(exchange.create_market_order, symbol, exit_side, qty, params={"reduceOnly": True})
            safe_msg = f"✅ [{symbol}] 긴급 시장가 청산 성공. 포지션 롤백 완료."
            logging.info(safe_msg)
            send_n8n(safe_msg)
        except Exception as emergency_e:
            crit_msg = f"❌ [{symbol}] 긴급 청산 실패! 즉시 수동 청산 필요. 에러: {emergency_e}"
            logging.critical(crit_msg)
            send_n8n(crit_msg)

        raise RuntimeError("TP/SL 설정 실패로 인한 포지션 보호 조치 발동")

# =========================================================
# 6) 국면별 시그널 로직 (요청사항 반영)
# =========================================================
def evaluate_signal_by_regime(
    symbol: str,
    df: pd.DataFrame,
    regime: str,
) -> Tuple[Optional[str], float, float, float, float]:
    """
    Returns:
      (signal, entry_price, tp_price, sl_price, mid_line)

    규칙:
      - TREND_UP  : 확정봉 저가(l)가 mid 이하 터치 -> LONG (entry=mid)
      - TREND_DOWN: 확정봉 고가(h)가 mid 이상 터치 -> SHORT (entry=mid)
      - RANGING   : 확정봉 고가가 upper 이상 -> SHORT (entry=upper)
                   확정봉 저가가 lower 이하 -> LONG (entry=lower)
      - TP/SL: 진입가 기준 SL 1%, TP 1.5%
    """
    if len(df) < WINDOW_SIZE + 5:
        return None, 0.0, 0.0, 0.0, 0.0

    i = -2  # 신호 판단은 확정봉 기준

    # 중심선(mid): 5m SMA(WINDOW_SIZE)
    mid_series = df["c"].rolling(WINDOW_SIZE).mean()
    mid = float(mid_series.iloc[i])
    if pd.isna(mid) or mid <= 0:
        return None, 0.0, 0.0, 0.0, 0.0

    high = float(df["h"].iloc[i])
    low = float(df["l"].iloc[i])

    upper = mid * (1 + ENVELOPE_PERCENT)
    lower = mid * (1 - ENVELOPE_PERCENT)

    signal: Optional[str] = None

    # 국면별 진입 신호 확인
    if regime == "TREND_UP":
        if low <= mid:
            signal = "LONG"
    elif regime == "TREND_DOWN":
        if high >= mid:
            signal = "SHORT"
    else:
        hit_upper = high >= upper
        hit_lower = low <= lower

        if hit_upper and not hit_lower:
            signal = "SHORT"
        elif hit_lower and not hit_upper:
            signal = "LONG"

    if not signal:
        return None, 0.0, 0.0, 0.0, mid

    # ⭐ 핵심 수정: TP/SL 계산의 기준을 '과거 타점'이 아닌 '현재 실시간 가격'으로 변경!
    # df["c"].iloc[-1] 은 아직 닫히지 않은 현재 캔들의 최신 가격입니다.
    current_price = float(df["c"].iloc[-1])

    sl_pct = 0.01
    tp_pct = 0.015

    # 실시간 가격 기준으로 위아래 TP/SL을 씌워줍니다 (-2021 에러 완벽 차단)
    if signal == "LONG":
        sl = current_price * (1 - sl_pct)
        tp = current_price * (1 + tp_pct)
    else:  # SHORT
        sl = current_price * (1 + sl_pct)
        tp = current_price * (1 - tp_pct)

    # 심볼별 가격 정밀도 적용
    current_price_precision = price_precision(symbol, current_price)
    tp = price_precision(symbol, tp)
    sl = price_precision(symbol, sl)

    # 수량(qty) 계산을 정확히 하기 위해 과거 기준가가 아닌 현재가를 리턴합니다.
    return signal, float(current_price_precision), float(tp), float(sl), float(mid)

# =========================================================
# 메인 및 종료 처리
# =========================================================
def handle_exit(signum, frame):
    """Ctrl+C 및 pm2 stop 시그널을 모두 받아 처리하는 단일 핸들러"""
    logging.info("🛑 종료 시그널 수신. 봇을 안전하게 종료합니다.")
    send_n8n("🛑 봇 종료: 트레이딩 엔진이 가동을 멈췄습니다.")
    sys.exit(0)

# =========================================================
# 7) 메인
# =========================================================
def main():
    # 윈도우(Ctrl+C)와 리눅스(pm2 stop) 종료 시그널을 하나의 핸들러로 연결
    sig.signal(sig.SIGINT, handle_exit)
    sig.signal(sig.SIGTERM, handle_exit)

    logging.info(f"🚀 다중 심볼 매매 엔진 가동: {', '.join(SYMBOLS)}")
    send_n8n(f"🤖 봇 가동 시작: 감시 종목 {len(SYMBOLS)}개")

    # 레버리지 설정
    for sym in SYMBOLS:
        set_symbol_leverage(sym, LEVERAGE)
        time.sleep(0.5)

    prev_pos_size: Dict[str, float] = {s: 0.0 for s in SYMBOLS}

    while True:
        try:
            equity = get_futures_usdt_equity()
            if equity <= 0:
                logging.warning("선물 USDT 잔고를 가져오지 못했습니다.")
            else:
                logging.info(f"💰 Futures Equity(USDT): {equity:.2f}")

            pos_map = fetch_positions_map(SYMBOLS)
            open_count = sum(1 for s in SYMBOLS if pos_map[s][0] > 0)

            for symbol in SYMBOLS:
                pos_size, pos_side = pos_map[symbol]

                # 포지션 종료 감지(잔여 주문 취소)
                if prev_pos_size[symbol] > 0 and pos_size == 0:
                    logging.info(f"🔔 [{symbol}] 포지션 종료 감지! 잔여 미체결 주문 강제 취소를 시도합니다.")
                    try:
                        # 1단계: CCXT 기본 취소 (일반 지정가 주문 클리어)
                        try:
                            safe_call(exchange.cancel_all_orders, symbol)
                        except: pass
                        
                        # 2단계: 최신 바이낸스 네이티브 API 폭격 (분리된 TP/SL 알고리즘 주문 싹쓸이)
                        market_id = exchange.market_id(symbol) # 예: 'BTCUSDT'
                        
                        # 조건부(TP/SL) 주문 전용 삭제 API 호출 (fapiPrivate_delete_algoopenorders)
                        if hasattr(exchange, 'fapiPrivate_delete_algoopenorders'):
                            safe_call(exchange.fapiPrivate_delete_algoopenorders, {'symbol': market_id})
                        elif hasattr(exchange, 'fapiPrivateDeleteAlgoOpenOrders'):
                            safe_call(exchange.fapiPrivateDeleteAlgoOpenOrders, {'symbol': market_id})
                            
                        time.sleep(0.5) # 서버 반영 대기
                        
                        logging.info(f"🧹 [{symbol}] 네이티브 Algo API 폭격으로 찌꺼기 주문(TP/SL) 싹쓸이 완료")
                        send_n8n(f"🧹 [{symbol}] 포지션 종료: 모든 예약 주문 강제 삭제 완료")
                            
                    except Exception as e:
                        logging.error(f"[{symbol}] 잔여 주문 강제 취소 실패: {e}")

                prev_pos_size[symbol] = pos_size

                if pos_size > 0:
                    continue
                if open_count >= MAX_OPEN_POSITIONS:
                    continue

                # 15분봉 데이터 로드
                ohlcv = safe_call(exchange.fetch_ohlcv, symbol, timeframe=VOL_TF, limit=WINDOW_SIZE + 250)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

                # ✅ 국면 판별 (아직 “필터”가 아니라 “엔진 선택”에 사용)
                regime = check_market_regime(symbol)

                # ✅ 국면별 시그널/TP/SL 계산
                signal, entry_price, tp_price, sl_price, mid_line = evaluate_signal_by_regime(symbol, df, regime)
                if not signal:
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 기존 필터 평가 (원하면 차후 국면과 중복되는 필터는 정리 가능)
                trend_status = check_trend_filter(symbol)
                vol_ok = check_volatility_filter(df)
                volume_ok = check_volume_filter(df)

                trend_ok = (
                    trend_status == "BOTH"
                    or (signal == "LONG" and trend_status == "LONG_ONLY")
                    or (signal == "SHORT" and trend_status == "SHORT_ONLY")
                )

                logging.info(
                    f"🔍 [{symbol}] regime={regime} signal={signal} entry_ref={entry_price:.4f} mid={mid_line:.4f} "
                    f"TP={tp_price:.4f} SL={sl_price:.4f} | trend={trend_status} vol={vol_ok} volume={volume_ok}"
                )

                if not (trend_ok and vol_ok and volume_ok):
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 수량 계산 (리스크 기반: entry ~ SL 거리)
                qty = calc_position_size(symbol, entry_price, sl_price, equity)
                if qty <= 0:
                    logging.warning(f"[{symbol}] 수량 계산 실패(0). 잔고/최소주문/SL거리 확인.")
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 주문 실행
                logging.info(f"🌟 [{symbol}] 조건 통과! regime={regime} qty={qty} 주문 실행")
                try:
                    place_entry_and_brackets(symbol, signal, qty, tp_price, sl_price)
                    msg = (
                        f"🚀 [{symbol}] {signal} 진입 ({regime})\n"
                        f"qty: {qty}\nref_entry: {entry_price}\nTP: {tp_price}\nSL: {sl_price}"
                    )
                    send_n8n(msg)
                    open_count += 1
                except Exception as e:
                    logging.error(f"[{symbol}] 주문 프로세스 에러: {e}")
                    send_n8n(f"❌ [{symbol}] 주문/방어 프로세스 에러: {e}")

                time.sleep(SYMBOL_SLEEP_SEC)

            time.sleep(LOOP_SLEEP_SEC)

        except Exception as e:
            logging.error(f"❌ 메인 루프 에러: {e}")
            time.sleep(ERROR_SLEEP_SEC)


if __name__ == "__main__":
    main()