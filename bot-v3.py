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

# =========================================================
# 0) 로깅/환경변수
# =========================================================
# - 로그는 콘솔 + 파일(trade.log)에 동시에 기록
# - 운영 시 문제 발생(주문 실패/네트워크/체결 불일치 등) 추적을 위해 타임스탬프 포함
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trade.log", encoding="utf-8"), logging.StreamHandler()],
)

# .env 파일에서 키 로딩 (운영환경에서 소스에 키를 박아 넣지 않기 위함)
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# =========================================================
# 1) 기본 설정
# =========================================================
# ⭐ 다중 심볼 리스트 (원하는 코인을 자유롭게 추가/삭제)
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# 선물 레버리지 (심볼별 set_leverage로 적용)
LEVERAGE = 5

# -------------------------
# 자금/리스크 관리 파라미터
# -------------------------
# BET_RATE:
# - "노출(Notional) 상한" 역할
# - equity * BET_RATE * LEVERAGE 까지만 포지션의 명목가치를 허용 (과도한 포지션 확대 방지)
BET_RATE = 0.05

# RISK_PER_TRADE:
# - "손실 리스크" 기준(가장 중요한 생존 파라미터)
# - 1회 트레이드에서 계좌의 몇 %까지 손실을 허용할지 (0.2~1% 권장)
# - 포지션 수량은 이 리스크를 기준으로 SL 거리(진입가-손절가)에 의해 결정됨
RISK_PER_TRADE = 0.005

# 동시 포지션 제한:
# - 심볼을 여러 개 감시하더라도 동시에 여러 포지션을 잡으면 리스크가 급증
# - 운영 안정성을 위해 기본 1 권장 (원하면 늘리되 리스크/손절/상관관계 고려 필수)
MAX_OPEN_POSITIONS = 1

# -------------------------
# 전략 파라미터
# -------------------------
# WINDOW_SIZE:
# - 5분봉 SMA(60) = 300분(5시간)의 중심선
WINDOW_SIZE = 20

# ENVELOPE_PERCENT:
# - TP/SL 기준을 "중심선(mid_line) ± x%"로 둔 단순한 밴드
# - 전략 의도: 중심선 교차 후 중심선 기반 밴드까지 도달하면 익절/손절
ENVELOPE_PERCENT = 0.01

# -------------------------
# 필터 스위치 (운영 중 빠르게 On/Off)
# -------------------------
ENABLE_TREND_FILTER = True
ENABLE_VOLATILITY_FILTER = False
ENABLE_VOLUME_FILTER = False

# -------------------------
# 필터 기준
# -------------------------
# 추세 필터:
# - 1시간봉 SMA20 기준으로 현재(확정봉) 종가가 위/아래인지 판단
TREND_TF = "1h"
TREND_SMA = 20

# 변동성(ATR) 필터:
# - ATR(14)을 계산한 뒤 ATR/가격(%)이 최소치 이상일 때만 거래
VOL_TF = "5m"
VOL_LOOKBACK = 14
VOL_MIN_ATR_PCT = 0.10

# 거래량 필터:
# - 확정봉 거래량이 최근 평균 대비 스파이크(배수) 이상일 때만 거래
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.5

# -------------------------
# 실행 루프 주기
# -------------------------
# - 심볼별 처리를 너무 빠르게 돌리면 API RateLimit/네트워크 에러 증가
# - 너무 느리면 신호를 놓칠 수 있음 (전략/운영 환경에 맞춰 조절)
SYMBOL_SLEEP_SEC = 1.5
LOOP_SLEEP_SEC = 30
ERROR_SLEEP_SEC = 10

# =========================================================
# 2) 거래소 초기화
# =========================================================
# - Binance USDT-M Futures 사용 (ccxt option defaultType='future')
# - enableRateLimit=True: ccxt 내부에서 rate limit을 어느 정도 조절하지만,
#   실전에서는 네트워크/거래소 오류가 여전히 발생하므로 safe_call로 보강
exchange = ccxt.binance(
    {
        "apiKey": API_KEY,
        "secret": SECRET_KEY,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
)

# 시작 시 마켓 메타를 로드:
# - min amount / min notional / tick size 등 거래 제약을 계산할 때 필요
exchange.load_markets()

# =========================================================
# 3) 유틸: 안전 호출/알림/정밀도/최소주문
# =========================================================
def send_n8n(msg: str) -> None:
    """
    N8N 웹훅 알림 (운영 모니터링)
    - URL이 없으면 조용히 무시 (로컬 테스트/비활성화 상황 고려)
    - 실패/에러도 로깅만 하고 봇은 계속 동작 (알림 장애로 매매 중단 방지)
    """
    if not N8N_WEBHOOK_URL:
        return
    try:
        r = requests.post(N8N_WEBHOOK_URL, json={"message": msg}, timeout=5)
        if r.status_code >= 400:
            logging.warning(f"N8N 응답 오류({r.status_code}): {msg}")
    except Exception as e:
        logging.warning(f"N8N 전송 실패: {e} | msg={msg}")


def safe_call(fn, *args, retries: int = 3, base_sleep: float = 0.6, **kwargs):
    """
    거래소/네트워크 오류에 대비한 재시도 래퍼.
    - NetworkError/Timeout: 지수 백오프로 재시도
    - ExchangeError: 일시적 오류일 수도 있어 소수 재시도 후 실패 처리
    """
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
    """
    선물 계정 USDT 기준 equity 추정치.
    - ccxt 통합 구조는 계정/설정에 따라 필드가 다를 수 있어 여러 경로를 시도
    - 운영 환경에서 0이 자주 나오면, 계정 타입/권한/ccxt 버전/파라미터(type='future') 점검 필요
    """
    bal = safe_call(exchange.fetch_balance, params={"type": "future"})
    if isinstance(bal, dict):
        # 1) 통합(unified) USDT 필드
        usdt = bal.get("USDT")
        if isinstance(usdt, dict):
            total = usdt.get("total")
            free = usdt.get("free")
            if total is not None:
                return float(total)
            if free is not None:
                return float(free)

        # 2) 거래소 raw info 기반 (바이낸스 선물 계정에서 자주 사용되는 후보 필드)
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
    """
    심볼별 최소 주문 노셔널(USDT) 추정.
    - 가능한 경우: market['limits']['cost']['min']
    - 없으면: market['info']에서 흔한 키들을 탐색
    - 최후: 보수적으로 5 USDT 반환 (심볼에 따라 다를 수 있으므로 운영 시 검증 권장)
    """
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
    """거래소 수량 정밀도(stepSize)에 맞춰 반올림/절삭"""
    return float(exchange.amount_to_precision(symbol, amount))


def price_precision(symbol: str, price: float) -> float:
    """거래소 가격 정밀도(tickSize)에 맞춰 반올림/절삭"""
    return float(exchange.price_to_precision(symbol, price))


# =========================================================
# 4) 지표/필터
# =========================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(True Range 기반) 계산.
    - 단순 (H-L) 변동성보다 갭(전봉 종가 대비)을 포함하므로 변동성 측정이 더 안정적
    """
    high, low, close = df["h"], df["l"], df["c"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


@dataclass
class TrendCacheItem:
    # 추세 필터는 1시간봉 기반이므로 매 루프마다 호출할 필요가 없음
    # → TTL 동안 캐싱하여 API 비용/지연을 크게 줄임
    ts: float
    mode: str  # 'LONG_ONLY' | 'SHORT_ONLY' | 'BOTH'


trend_cache: Dict[str, TrendCacheItem] = {}
TREND_CACHE_TTL_SEC = 300  # 5분 캐시


def check_trend_filter(symbol: str) -> str:
    """
    추세 필터:
    - 1시간봉 SMA20 기준
    - 확정봉(-2) 종가로 판단하여 "봉 진행 중 흔들림"을 제거
    """
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
        sma = df["c"].rolling(TREND_SMA).mean().iloc[-2]   # 확정봉 기준
        close = df["c"].iloc[-2]
        mode = "LONG_ONLY" if close > sma else "SHORT_ONLY"

    trend_cache[symbol] = TrendCacheItem(ts=now, mode=mode)
    return mode


def check_volatility_filter(df_5m: pd.DataFrame) -> bool:
    """
    변동성 필터:
    - 확정봉(-2) 기준 ATR/Price(%)가 최소치 이상인지 확인
    - 변동성 너무 낮으면 whipsaw가 늘거나 기대수익이 줄어드는 구간을 회피
    """
    if not ENABLE_VOLATILITY_FILTER:
        return True
    atr = compute_atr(df_5m, period=VOL_LOOKBACK)
    atr_last = atr.iloc[-2]            # 확정봉 ATR
    price_last = df_5m["c"].iloc[-2]   # 확정봉 종가
    if pd.isna(atr_last) or price_last <= 0:
        return False
    return ((atr_last / price_last) * 100.0) >= VOL_MIN_ATR_PCT


def check_volume_filter(df_5m: pd.DataFrame) -> bool:
    """
    거래량 필터:
    - 진행 중 봉(-1)은 거래량이 계속 변하므로 사용하지 않음
    - 확정봉(-2) 거래량이 최근 평균 대비 스파이크인지 체크
    """
    if not ENABLE_VOLUME_FILTER:
        return True
    if len(df_5m) < VOLUME_LOOKBACK + 3:
        return False

    vols = df_5m["v"]
    baseline = vols.iloc[-(VOLUME_LOOKBACK + 2) : -2].mean()  # 과거 확정봉 평균
    last_vol = vols.iloc[-2]                                  # 마지막 확정봉

    if baseline <= 0:
        return False
    return last_vol >= baseline * VOLUME_SPIKE_MULT


# =========================================================
# 5) 포지션/주문 관리
# =========================================================
def fetch_positions_map(symbols) -> Dict[str, Tuple[float, Optional[str]]]:
    """
    포지션 맵 구성:
    - 숏 포지션은 contracts가 음수일 수 있으므로 abs()로 보유 여부 판단 (중요!)
    - 반환: {symbol: (pos_size_abs, 'LONG'/'SHORT'/None)}
    """
    positions = safe_call(exchange.fetch_positions, symbols)
    out = {s: (0.0, None) for s in symbols}

    for p in positions:
        psym = p.get("symbol", "")
        for s in symbols:
            # ccxt가 'BTC/USDT:USDT'로 주는 경우가 있어 포함 비교
            if s in psym:
                contracts = None

                # 거래소/ccxt 버전마다 필드명이 다를 수 있어 후보를 순회
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
    """
    심볼별 레버리지 설정.
    - 실패 시 해당 심볼은 거래 제외가 더 안전 (여기서는 로그만 남김)
    """
    try:
        safe_call(exchange.set_leverage, lev, symbol)
        logging.info(f"[{symbol}] 레버리지 {lev}배 설정 완료")
    except Exception as e:
        logging.error(f"[{symbol}] 레버리지 설정 실패: {e}")


def calc_position_size(symbol: str, entry_price: float, stop_price: float, equity_usdt: float) -> float:
    """
    리스크 기반 포지션 사이징:
    1) risk_usdt = equity * RISK_PER_TRADE
    2) qty_raw = risk_usdt / |entry - stop|
       - 손절까지 이동하면 대략 risk_usdt 손실이 되도록 설계
    3) qty_cap: 노출 상한(equity * BET_RATE * LEVERAGE)을 넘지 않도록 캡
    4) 최소주문(min_notional/min_amount) 및 정밀도 보정
    """
    if equity_usdt <= 0:
        return 0.0

    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0:
        return 0.0

    risk_usdt = equity_usdt * RISK_PER_TRADE
    raw_qty = risk_usdt / stop_dist

    # 노출 캡(명목가치 상한)
    max_notional = equity_usdt * BET_RATE * LEVERAGE
    qty_cap = max_notional / entry_price
    qty = min(raw_qty, qty_cap)

    # 최소 주문 조건(노셔널/수량)
    min_notional = get_market_min_notional(symbol)
    min_qty_from_notional = min_notional / entry_price

    m = exchange.market(symbol)
    min_amount = m.get("limits", {}).get("amount", {}).get("min", 0) or 0
    qty = max(qty, float(min_qty_from_notional), float(min_amount))

    # 거래소 수량 정밀도 적용
    qty = amount_precision(symbol, qty)

    # 정밀도 절삭으로 인해 min_notional 미달이 생길 수 있어 약간 보정
    if qty * entry_price < min_notional:
        qty = amount_precision(symbol, qty + (min_qty_from_notional * 0.2))

    return float(qty)


def place_entry_and_brackets(symbol: str, signal: str, qty: float, tp: float, sl: float) -> None:
    """
    ✅ Orphaned Position(고아 포지션) 방어 로직

    문제 상황:
    - 시장가 진입은 성공했는데,
    - 네트워크/거래소 오류로 TP/SL 주문이 등록되지 않으면,
      포지션이 보호(손절/익절) 없이 '노출' 상태로 남는 최악의 상황이 발생할 수 있음.

    해결:
    - [1단계] 진입(시장가)
    - [2단계] TP/SL 예약
      -> 여기서 실패하면 즉시 반대방향 reduceOnly 시장가로 긴급 청산(Emergency Close)하여 롤백 시도
    """
    side = "buy" if signal == "LONG" else "sell"
    exit_side = "sell" if side == "buy" else "buy"

    # [1단계] 진입 주문 (시장가)
    try:
        safe_call(exchange.create_market_order, symbol, side, qty)
    except Exception as e:
        # 진입 자체가 실패하면 포지션이 없으므로 보호조치(청산)가 필요 없음
        raise RuntimeError(f"시장가 진입 주문 실패: {e}")

    # [2단계] 익절/손절(TP/SL) 주문 예약
    try:
        safe_call(
            exchange.create_order,
            symbol,
            "TAKE_PROFIT_MARKET",
            exit_side,
            qty,
            None,
            params={"stopPrice": tp, "reduceOnly": True},
        )
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
        # TP/SL 등록 실패 → 고아 포지션 위험 → 긴급 청산으로 롤백
        alert_msg = f"🚨 [{symbol}] TP/SL 예약 실패! 고아 포지션 위험으로 긴급 청산을 시도합니다. 에러: {e}"
        logging.error(alert_msg)
        send_n8n(alert_msg)

        try:
            # 방금 진입한 수량만큼 반대 방향 reduceOnly 시장가 청산
            safe_call(exchange.create_market_order, symbol, exit_side, qty, params={"reduceOnly": True})
            safe_msg = f"✅ [{symbol}] 긴급 시장가 청산 성공. 포지션 롤백 완료."
            logging.info(safe_msg)
            send_n8n(safe_msg)
        except Exception as emergency_e:
            # 긴급 청산도 실패 → 즉시 수동 조치 필요
            crit_msg = (
                f"❌ [{symbol}] 긴급 청산 완전 실패! 즉시 바이낸스 앱에서 수동 청산하세요! "
                f"에러: {emergency_e}"
            )
            logging.critical(crit_msg)
            send_n8n(crit_msg)

        # 상위 로직에서 해당 트레이드 실패를 인지하도록 예외 발생
        raise RuntimeError("TP/SL 설정 에러로 인한 포지션 보호 조치 발동")


# =========================================================
# 6) 메인 전략 로직
# =========================================================
def evaluate_signal(df: pd.DataFrame) -> Tuple[Optional[str], float, float]:
    """
    ✅ (리뷰 반영) 중복 연산 제거(DRY) + 성능 개선

    기존:
    - evaluate_signal이 "LONG"/"SHORT"만 반환
    - main 루프에서 df['mid'] rolling 계산을 또 수행 → 불필요한 오버헤드

    변경:
    - rolling 계산을 evaluate_signal에서 1회 수행
    - (signal, mid_line, entry_price)를 한 번에 반환하여 main에서 재계산 제거

    주의:
    - 신호는 진행 중 봉(-1)이 아닌, 확정봉(-2, -3) 기준으로 판단
      (봉 마감 전 발생했다가 사라지는 '가짜 교차'를 줄이기 위함)
    """
    if len(df) < WINDOW_SIZE + 5:
        return None, 0.0, 0.0

    # 경고 방지를 위해 df 자체에 바로 저장하지 않고 Series로 계산
    rolling_mid = df["c"].rolling(WINDOW_SIZE).mean()

    # 확정봉 2개를 사용:
    # - prev: -3
    # - curr: -2 (가장 최근에 "마감된" 봉)
    mid_prev = rolling_mid.iloc[-3]
    mid_curr = rolling_mid.iloc[-2]

    c_prev = df["c"].iloc[-3]
    c_curr = df["c"].iloc[-2]

    # 데이터 부족/NaN 방어
    if pd.isna(mid_prev) or pd.isna(mid_curr):
        return None, 0.0, 0.0

    # 중심선 교차 판단
    signal = None
    if c_prev <= mid_prev and c_curr > mid_curr:
        signal = "LONG"
    elif c_prev >= mid_prev and c_curr < mid_curr:
        signal = "SHORT"

    # entry_price:
    # - 전략상 "확정봉 종가"를 진입 기준으로 사용
    # - 실제 시장가 체결은 슬리피지로 달라질 수 있음(운영 시 고려)
    return signal, float(mid_curr), float(c_curr)


def main():
    logging.info(f"🚀 다중 심볼 매매 엔진 가동: {', '.join(SYMBOLS)}")
    send_n8n(f"🤖 봇 가동 시작: 감시 종목 {len(SYMBOLS)}개")

    # -----------------------------------------------------
    # 심볼별 레버리지 일괄 설정
    # -----------------------------------------------------
    for sym in SYMBOLS:
        set_symbol_leverage(sym, LEVERAGE)
        time.sleep(0.5)

    # -----------------------------------------------------
    # 이전 포지션 수량 추적:
    # - 포지션이 0으로 바뀌는 순간을 감지하여 잔여 주문 취소
    # - 부분청산/증감 추적까지 확장하려면 pos_size 변화량도 기록 가능
    # -----------------------------------------------------
    prev_pos_size: Dict[str, float] = {s: 0.0 for s in SYMBOLS}

    while True:
        try:
            # 1) 선물 계정 잔고(Equity) 읽기
            equity = get_futures_usdt_equity()
            if equity <= 0:
                logging.warning("선물 USDT 잔고를 가져오지 못했습니다.")
            else:
                logging.info(f"💰 Futures Equity(USDT): {equity:.2f}")

            # 2) 포지션 맵 업데이트
            pos_map = fetch_positions_map(SYMBOLS)

            # 3) 동시 포지션 수 계산(리스크 제한)
            open_count = sum(1 for s in SYMBOLS if pos_map[s][0] > 0)

            # 4) 심볼별 전략 실행
            for symbol in SYMBOLS:
                pos_size, pos_side = pos_map[symbol]

                # 포지션 종료 감지 → 남아있는 TP/SL 등 미체결 주문 정리
                if prev_pos_size[symbol] > 0 and pos_size == 0:
                    logging.info(f"🔔 [{symbol}] 포지션 종료 감지! 잔여 미체결 주문을 일괄 취소합니다.")
                    try:
                        # 1. CCXT 버그 방지: 현재 열려있는 모든 주문(TP/SL 포함)을 직접 조회
                        open_orders = safe_call(exchange.fetch_open_orders, symbol)
                        
                        # 2. 주문이 존재하면 반복문을 돌며 하나씩 확실하게 취소 (Kill)
                        if open_orders:
                            for order in open_orders:
                                safe_call(exchange.cancel_order, order['id'], symbol)
                            
                            logging.info(f"🧹 [{symbol}] 잔여 주문 정리 완료 ({len(open_orders)}개 취소)")
                            send_n8n(f"🧹 [{symbol}] 포지션 종료: 미체결 주문 {len(open_orders)}개 정리 완료")
                        else:
                            logging.info(f"🧹 [{symbol}] 취소할 잔여 주문이 없습니다.")
                            
                    except Exception as e:
                        logging.error(f"[{symbol}] 잔여 주문 취소 실패: {e}")

                prev_pos_size[symbol] = pos_size

                # 포지션 보유 중이면 신호/주문 생성하지 않음
                if pos_size > 0:
                    continue

                # 동시 포지션 제한 초과 시 신규 진입 금지
                if open_count >= MAX_OPEN_POSITIONS:
                    continue

                # 5) 5분봉 데이터 로드
                # - limit은 WINDOW_SIZE + 여유분(필터/ATR 계산용)
                ohlcv = safe_call(exchange.fetch_ohlcv, symbol, timeframe=VOL_TF, limit=WINDOW_SIZE + 250)
                df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])

                # 6) 신호 및 핵심 지표 평가 (✅ 중복 연산 제거됨)
                signal, mid_line, entry_price = evaluate_signal(df)

                if not signal:
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 7) TP/SL 가격 설정 (중심선 밴드 기반)
                upper = price_precision(symbol, mid_line * (1 + ENVELOPE_PERCENT))
                lower = price_precision(symbol, mid_line * (1 - ENVELOPE_PERCENT))

                tp_price = upper if signal == "LONG" else lower
                sl_price = lower if signal == "LONG" else upper

                # 8) 필터 평가
                trend_status = check_trend_filter(symbol)
                vol_ok = check_volatility_filter(df)
                volume_ok = check_volume_filter(df)

                trend_ok = (
                    trend_status == "BOTH"
                    or (signal == "LONG" and trend_status == "LONG_ONLY")
                    or (signal == "SHORT" and trend_status == "SHORT_ONLY")
                )

                logging.info(
                    f"🔍 [{symbol}] 신호={signal} entry_ref={entry_price:.4f} mid={mid_line:.4f} "
                    f"TP={tp_price:.4f} SL={sl_price:.4f} | trend={trend_status} vol={vol_ok} volume={volume_ok}"
                )

                # 9) 필터 불통과 시 스킵
                if not (trend_ok and vol_ok and volume_ok):
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 10) 수량 계산(리스크 기반)
                qty = calc_position_size(symbol, entry_price, sl_price, equity)
                if qty <= 0:
                    logging.warning(f"[{symbol}] 수량 계산 실패(0). 잔고/최소주문/SL거리 확인.")
                    time.sleep(SYMBOL_SLEEP_SEC)
                    continue

                # 11) 주문 실행 + 방어 로직
                logging.info(f"🌟 [{symbol}] 조건 통과! qty={qty} 주문 실행")
                try:
                    place_entry_and_brackets(symbol, signal, qty, tp_price, sl_price)

                    # 알림 (진입 성공 + TP/SL 예약 성공)
                    msg = (
                        f"🚀 [{symbol}] {signal} 진입\n"
                        f"qty: {qty}\nref_entry: {entry_price}\nTP: {tp_price}\nSL: {sl_price}"
                    )
                    send_n8n(msg)

                    # 신규 포지션 1개 추가(동시 포지션 제한 반영)
                    open_count += 1

                except Exception as e:
                    # 여기로 오는 경우:
                    # - 진입 자체 실패
                    # - TP/SL 예약 실패로 긴급 청산 발동 후 예외 발생
                    logging.error(f"[{symbol}] 주문 프로세스 에러 발생: {e}")
                    send_n8n(f"❌ [{symbol}] 진입 또는 방어 프로세스 에러: {e}")

                time.sleep(SYMBOL_SLEEP_SEC)

            # 심볼 전체를 다 돌고 나서 루프 대기
            time.sleep(LOOP_SLEEP_SEC)

        except Exception as e:
            # 메인 루프 예외는 "봇 전체 다운"을 막기 위해 잡고 재시도
            logging.error(f"❌ 메인 루프 에러: {e}")
            time.sleep(ERROR_SLEEP_SEC)


if __name__ == "__main__":
    main()