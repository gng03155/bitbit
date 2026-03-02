import os
import logging
import ccxt
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
SYMBOL = 'BTC/USDT'
LEVERAGE = 5
MIN_TEST_AMOUNT = 0.002 

exchange = ccxt.binance({
    'apiKey': API_KEY, 'secret': SECRET_KEY,
    'enableRateLimit': True, 'options': {'defaultType': 'future'}
})

def run_test():
    try:
        # 1. 레버리지 설정
        exchange.set_leverage(LEVERAGE, SYMBOL)
        
        # 2. 현재가 조회
        ticker = exchange.fetch_ticker(SYMBOL)
        curr_price = ticker['last']
        
        logging.info(f"🔔 최종 TP/SL 테스트 시작: {SYMBOL}")

        # 3. 시장가 매수 (이미 성공했으므로 한 번 더 테스트)
        order = exchange.create_market_buy_order(SYMBOL, MIN_TEST_AMOUNT)
        logging.info(f"✅ 매수 성공! ID: {order['id']}")

        # 4. TP/SL 가격 계산
        tp_price = float(exchange.price_to_precision(SYMBOL, curr_price * 1.01))
        sl_price = float(exchange.price_to_precision(SYMBOL, curr_price * 0.99))

        # 5. 익절/손절 주문 (params 안에 stopPrice와 triggerPrice를 명시적으로 전달)
        # TAKE_PROFIT_MARKET 주문
        exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', MIN_TEST_AMOUNT, params={
            'stopPrice': tp_price,
            'triggerPrice': tp_price,  # 일부 버전 대응
            'reduceOnly': True
        })
        
        # STOP_MARKET 주문
        exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', MIN_TEST_AMOUNT, params={
            'stopPrice': sl_price,
            'triggerPrice': sl_price, # 일부 버전 대응
            'reduceOnly': True
        })
        
        logging.info(f"✅ TP({tp_price}) 및 SL({sl_price}) 예약 완료")
        logging.info("🎉 모든 로직이 완벽하게 작동합니다!")

    except Exception as e:
        logging.error(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    run_test()