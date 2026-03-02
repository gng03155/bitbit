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

SYMBOL = 'BTC/USDT'
LEVERAGE = 5
BET_RATE = 0.1
WINDOW_SIZE = 60
ENVELOPE_PERCENT = 0.01

# [필터 스위치]
ENABLE_TREND_FILTER = True
ENABLE_ATR_FILTER = True
ENABLE_VOLUME_FILTER = True

exchange = ccxt.binance({
    'apiKey': API_KEY, 'secret': SECRET_KEY,
    'enableRateLimit': True, 'options': {'defaultType': 'future'}
})

# ==========================================
# 2. 유틸리티 및 필터 함수
# ==========================================
def send_n8n(msg):
    try:
        requests.post(N8N_WEBHOOK_URL, json={"message": msg}, timeout=5)
    except:
        print(f"N8N 웹훅 전송 실패: {msg}")
        pass

def get_amount(curr_price):
    balance = exchange.fetch_balance()
    usdt_free = balance.get('USDT', {}).get('free', 0)
    amount = (usdt_free * BET_RATE * LEVERAGE) / curr_price
    if (amount * curr_price) < 105:
        amount = 105 / curr_price
    return float(exchange.amount_to_precision(SYMBOL, amount))

def check_trend_filter():
    if not ENABLE_TREND_FILTER:
        return 'BOTH'
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=60)
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
    logging.info("🚀 양방향 매매 엔진 가동 (잔여 주문 취소 적용, 쿨다운 제거)")
    send_n8n("🤖 봇 가동 시작: 3중 필터 및 자동 정리 기능 적용")
    
    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except Exception as e:
        logging.error(f"레버리지 설정 오류: {e}")

    # 이전 루프의 포지션 유무 기억
    was_in_position = False 

    while True:
        try:
            # 1. 포지션 조회 (정확한 수량 체크)
            positions = exchange.fetch_positions([SYMBOL])
            has_position = False
            
            for p in positions:
                if float(p.get('contracts', 0)) > 0:
                    has_position = True
                    break

            # 2. 포지션 종료 감지 및 잔여 주문(TP/SL) 자동 취소
            if was_in_position == True and has_position == False:
                logging.info("🔔 포지션 종료 감지! 남아있는 미체결 주문을 모두 취소합니다.")
                try:
                    exchange.cancel_all_orders(SYMBOL)
                    logging.info("🧹 잔여 주문 정리 완료")
                    send_n8n("🧹 포지션 종료: 미체결 주문 정리 완료")
                except Exception as e:
                    logging.error(f"잔여 주문 취소 실패: {e}")

            # 상태 업데이트
            was_in_position = has_position

            # 3. 차트 데이터 로드 및 지표 계산
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=300)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['mid'] = df['c'].rolling(WINDOW_SIZE).mean()
            
            curr_price = df['c'].iloc[-1]
            prev_price = df['c'].iloc[-2]
            mid_line = df['mid'].iloc[-1]
            
            upper = float(exchange.price_to_precision(SYMBOL, mid_line * (1 + ENVELOPE_PERCENT)))
            lower = float(exchange.price_to_precision(SYMBOL, mid_line * (1 - ENVELOPE_PERCENT)))

            # 4. 매매 판단 및 실행
            if has_position:
                logging.info(f"🔒 현재 포지션 보유 중 (대기). 현재가: {curr_price}")
            else:
                logging.info(f"🔍 감시 - 현재가: {curr_price} | 1봉전: {prev_price} | 중심선: {mid_line:.2f}")
                signal = None
                
                # 기본 돌파 신호 확인 (골든/데드크로스)
                if prev_price <= mid_line < curr_price:
                    signal = 'LONG'
                elif curr_price < mid_line <= prev_price:
                    signal = 'SHORT'
                
                if signal:
                    trend_status = check_trend_filter()
                    is_volatile = check_atr_filter(df)
                    has_volume = check_volume_filter(df)
                    
                    logging.info(f"기본 신호: {signal} | 추세: {trend_status} | 변동성: {is_volatile} | 거래량: {has_volume}")
                    
                    trend_passed = (trend_status == 'BOTH') or (signal == 'LONG' and trend_status == 'LONG_ONLY') or (signal == 'SHORT' and trend_status == 'SHORT_ONLY')
                    
                    if trend_passed and is_volatile and has_volume:
                        logging.info("🌟 모든 필터 통과! 주문을 실행합니다.")
                        
                        amt = get_amount(curr_price)
                        side = 'buy' if signal == 'LONG' else 'sell'
                        exit_side = 'sell' if side == 'buy' else 'buy'
                        tp_price = upper if signal == 'LONG' else lower
                        sl_price = lower if signal == 'LONG' else upper
                        
                        # 주문 실행
                        exchange.create_market_order(SYMBOL, side, amt)
                        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', exit_side, amt, params={'stopPrice': tp_price, 'triggerPrice': tp_price, 'reduceOnly': True})
                        exchange.create_order(SYMBOL, 'STOP_MARKET', exit_side, amt, params={'stopPrice': sl_price, 'triggerPrice': sl_price, 'reduceOnly': True})
                        
                        msg = f"🚀 {signal} 진입 완료\n진입가: {curr_price}\n목표가: {tp_price}\n손절가: {sl_price}"
                        send_n8n(msg)
                    else:
                        logging.info(f"🚫 추세: {trend_status} | 변동성: {is_volatile} | 거래량: {has_volume} 미달로 진입 포기")

            time.sleep(60)

        except Exception as e:
            logging.error(f"❌ 루프 에러: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()