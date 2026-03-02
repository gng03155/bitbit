import os
import time
import logging
import pandas as pd
import ccxt
import requests
from dotenv import load_dotenv

# ==========================================
# 1. 기본 설정 및 스위치
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler("trade.log", encoding='utf-8'), logging.StreamHandler()])
load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL')

# ⭐ 다중 심볼 리스트 (원하는 코인을 자유롭게 추가/삭제하세요)
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

LEVERAGE = 5
BET_RATE = 0.05 # 전체 잔고의 10%를 각 종목당 할당 (자금 관리에 주의!)
WINDOW_SIZE = 60
ENVELOPE_PERCENT = 0.01

ENABLE_TREND_FILTER = True
ENABLE_ATR_FILTER = True
ENABLE_VOLUME_FILTER = True

exchange = ccxt.binance({
    'apiKey': API_KEY, 'secret': SECRET_KEY,
    'enableRateLimit': True, 'options': {'defaultType': 'future'}
})

# ==========================================
# 2. 유틸리티 및 필터 함수 (symbol 매개변수 추가)
# ==========================================
def send_n8n(msg):
    try:
        requests.post(N8N_WEBHOOK_URL, json={"message": msg}, timeout=5)
    except:
        print(f"N8N 웹훅 전송 실패: {msg}")
        pass

def get_amount(symbol, curr_price):
    balance = exchange.fetch_balance()
    usdt_free = balance.get('USDT', {}).get('free', 0)
    amount = (usdt_free * BET_RATE * LEVERAGE) / curr_price
    if (amount * curr_price) < 105:
        amount = 105 / curr_price
    return float(exchange.amount_to_precision(symbol, amount))

def check_trend_filter(symbol):
    if not ENABLE_TREND_FILTER:
        return 'BOTH'
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=60)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    sma_50 = df['c'].rolling(50).mean().iloc[-1]
    curr_price = df['c'].iloc[-1]
    return 'LONG_ONLY' if curr_price > sma_50 else 'SHORT_ONLY'

def check_atr_filter(df_5m):
    if not ENABLE_ATR_FILTER:
        return True
    df_5m['range_pct'] = (df_5m['h'] - df_5m['l']) / df_5m['c'] * 100
    avg_volatility = df_5m['range_pct'].rolling(10).mean().iloc[-1]
    return avg_volatility >= 0.1

def check_volume_filter(df_5m):
    if not ENABLE_VOLUME_FILTER:
        return True
    avg_vol = df_5m['v'].iloc[-6:-1].mean()
    prev_vol = df_5m['v'].iloc[-2]
    curr_vol = df_5m['v'].iloc[-1]
    return (prev_vol > avg_vol * 1.5) or (curr_vol > avg_vol * 1.5)

# ==========================================
# 3. 메인 매매 로직
# ==========================================
def main():
    logging.info(f"🚀 다중 심볼 매매 엔진 가동: {', '.join(SYMBOLS)}")
    send_n8n(f"🤖 봇 가동 시작: 감시 종목 {len(SYMBOLS)}개")
    
    # 각 종목별 레버리지 일괄 설정
    for sym in SYMBOLS:
        try:
            exchange.set_leverage(LEVERAGE, sym)
            logging.info(f"[{sym}] 레버리지 {LEVERAGE}배 설정 완료")
            time.sleep(1) # API Rate Limit 방지
        except Exception as e:
            logging.error(f"[{sym}] 레버리지 설정 오류: {e}")

    # ⭐ 종목별 이전 포지션 상태를 딕셔너리로 관리 (초기값 False)
    was_in_position = {sym: False for sym in SYMBOLS}

    while True:
        try:
            # 1. 모든 종목의 포지션을 한 번의 API 호출로 가져옴 (효율성)
            positions = exchange.fetch_positions(SYMBOLS)
            
            # 현재 잡혀있는 포지션을 매핑 (예: {'BTC/USDT': True, 'ETH/USDT': False})
            active_positions = {sym: False for sym in SYMBOLS}
            for p in positions:
                for sym in SYMBOLS:
                    # CCXT가 반환하는 'BTC/USDT:USDT' 같은 형식 대응을 위해 in 연산자 사용
                    if sym in p.get('symbol', '') and float(p.get('contracts', 0)) > 0:
                        active_positions[sym] = True

            # 2. 각 심볼별로 개별 로직 실행 (Loop)
            for symbol in SYMBOLS:
                has_position = active_positions[symbol]

                # [잔여 주문 정리 로직] 포지션이 종료된 순간 감지
                if was_in_position[symbol] == True and has_position == False:
                    logging.info(f"🔔 [{symbol}] 포지션 종료 감지! 미체결 주문 취소")
                    try:
                        exchange.cancel_all_orders(symbol)
                        logging.info(f"🧹 [{symbol}] 잔여 주문 정리 완료")
                        send_n8n(f"🧹 [{symbol}] 포지션 종료: 미체결 주문 정리 완료")
                    except Exception as e:
                        logging.error(f"[{symbol}] 잔여 주문 취소 실패: {e}")

                # 상태 업데이트
                was_in_position[symbol] = has_position

                if has_position:
                    # 로그가 너무 길어지지 않게 대기 로그는 제외하거나 디버그 레벨로 관리해도 좋습니다.
                    # logging.info(f"🔒 [{symbol}] 현재 포지션 보유 중 (대기).")
                    continue

                # 3. 차트 데이터 로드 및 지표 계산
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=300)
                df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                df['mid'] = df['c'].rolling(WINDOW_SIZE).mean()
                
                curr_price = df['c'].iloc[-1]
                prev_price = df['c'].iloc[-2]
                mid_line = df['mid'].iloc[-1]
                
                upper = float(exchange.price_to_precision(symbol, mid_line * (1 + ENVELOPE_PERCENT)))
                lower = float(exchange.price_to_precision(symbol, mid_line * (1 - ENVELOPE_PERCENT)))

                logging.info(f"👀 [{symbol}] 감시중 - 현재가: {curr_price} | 중심선: {mid_line:.2f}")

                # 4. 매매 판단
                signal = None
                if prev_price <= mid_line < curr_price:
                    signal = 'LONG'
                elif curr_price < mid_line <= prev_price:
                    signal = 'SHORT'
                
                if signal:
                    logging.info(f"🔍 [{symbol}] 신호 포착 - 현재가: {curr_price} | 중심선: {mid_line:.2f}")
                    
                    trend_status = check_trend_filter(symbol)
                    is_volatile = check_atr_filter(df)
                    has_volume = check_volume_filter(df)
                    
                    logging.info(f"[{symbol}] {signal} | 추세: {trend_status} | 변동성: {is_volatile} | 거래량: {has_volume}")
                    
                    trend_passed = (trend_status == 'BOTH') or (signal == 'LONG' and trend_status == 'LONG_ONLY') or (signal == 'SHORT' and trend_status == 'SHORT_ONLY')
                    
                    if trend_passed and is_volatile and has_volume:
                        logging.info(f"🌟 [{symbol}] 필터 올패스! 주문 실행")
                        
                        amt = get_amount(symbol, curr_price)
                        side = 'buy' if signal == 'LONG' else 'sell'
                        exit_side = 'sell' if side == 'buy' else 'buy'
                        tp_price = upper if signal == 'LONG' else lower
                        sl_price = lower if signal == 'LONG' else upper
                        
                        exchange.create_market_order(symbol, side, amt)
                        exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', exit_side, amt, params={'stopPrice': tp_price, 'triggerPrice': tp_price, 'reduceOnly': True})
                        exchange.create_order(symbol, 'STOP_MARKET', exit_side, amt, params={'stopPrice': sl_price, 'triggerPrice': sl_price, 'reduceOnly': True})
                        
                        msg = f"🚀 [{symbol}] {signal} 진입\n단가: {curr_price}\nTP: {tp_price}\nSL: {sl_price}"
                        send_n8n(msg)

                # 심볼 간 처리 시 API 호출 제한 방지용 짧은 대기
                time.sleep(2) 

            # 모든 심볼 처리가 끝나면 1분 대기
            time.sleep(60)

        except Exception as e:
            logging.error(f"❌ 메인 루프 에러: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()