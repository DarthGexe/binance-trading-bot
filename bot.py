import datetime as dt
import time
import pytz
import pandas as pd
import numpy as np
from binance.client import Client

# API key and Secret
API_KEY = 'YOUR API KEY'
API_SECRET = 'YOUR API SECRET'

crypto = 'BTC'
ref = 'USDT'
symbol = crypto + ref
amount = 15

# Eligible periods for candlesticks 1m, 3m, 5m, 15m, 1h, etc.
period = '15m'

# Moving Average Periods
ema_f = 15
ema_s = 50

# Functions
def get_balance(balances, asset):
    for bal in balances:
        if bal.get('asset') == asset:
            return float(bal['free'])

def get_max_float_qty(step_size):
    max_float_qty = 0
    a = 10
    while step_size * a < 1:
      a = a*10**max_float_qty
      max_float_qty += 1
    return max_float_qty

def get_quantity(price, amount, min_qty, max_qty, max_float_qty):
    quantity = amount / price
    if (quantity < min_qty or quantity > max_qty):
        return False
    quantity = np.round(quantity, max_float_qty)
    return quantity

def crossover(ma_fast, ma_slow):
    if (ma_fast[0] < ma_slow[0] and ma_fast[1] >= ma_slow[1]):
        return True
    return False

def synchronize():
    candles = client.get_klines(symbol=symbol,interval=period,limit=1)
    timer = pd.to_datetime(float(candles[0][0]) * 1000000)
    start = timer + dt.timedelta(minutes=step)
    print('Synchronizing .....')
    while dt.datetime.now(dt.timezone.utc) < pytz.UTC.localize(start):
        time.sleep(1)
    time.sleep(2)
    print('Bot synchronized')

def parser(limit=ema_s+1):
    candles = client.get_klines(symbol=symbol,interval=period,limit=limit)
    df = pd.DataFrame(candles)
    df = df.drop([6, 7, 8, 9, 10, 11], axis=1)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df[['time', 'open', 'high', 'low', 'close', 'volume']] = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'] * 1000000)
    df['ema_s'] = df['close'].ewm(span=ema_s).mean()
    df['ema_f'] = df['close'].ewm(span=ema_f).mean()

    df['operation'] = np.nan

    return df

def operate(df):
    operation = ''
    price = df['close'].iloc[-1]
    if crossover(df.ema_f.values[-2:], df.ema_s.values[-2:]):
        quantity = get_quantity(price, amount, min_qty, max_qty, max_float_qty)
        balances = client.get_account()['balances']
        balance = get_balance(balances, ref)
        if not quantity:
            print('No Quantity available')
        elif balance <= amount:
            print(f'No {ref} to buy {crypto}')
        else:
            order = client.order_market_buy(
                symbol=symbol,
                quantity=quantity)
            print('BUY')
            operation = 'BUY'
    elif crossover(df.ema_s.values[-2:], df.ema_f.values[-2:]):
        quantity = get_quantity(price, amount, min_qty, max_qty, max_float_qty)
        balances = client.get_account()['balances']
        balance = get_balance(balances, crypto)
        if not quantity:
            print('No Quantity available')
        elif balance < quantity:
            print(f'No {crypto} to sell for {ref}')
        else:
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity)
            print('SELL')
            operation = 'SELL'
    return operation

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    client = Client(API_KEY, API_SECRET)
    info = client.get_symbol_info(symbol)
    min_qty = float(info['filters'][1].get('minQty'))
    step_size = float(info['filters'][1].get('stepSize'))
    max_qty = float(info['filters'][1].get('maxQty'))
    max_float_qty = get_max_float_qty(step_size)

    df = parser()

    step = int(period[:2]) if len(period) > 2 else int(period[0])
    if 'h' in period:
        step = step * 60

    synchronize()

    while True:
        temp = time.time()

        data_df = parser()
        data_df['operation'].iloc[-1] = operate(data_df.tail(2))
        df = pd.concat([df, data_df.tail(1)], ignore_index=True)

        if df.shape[0] > 10000:
            df = df.tail(10000)
        df.to_csv(f'./{symbol}_{period}.csv')

        delay = time.time() - temp
        idle = 60 * step - delay
        print(f'idle = {idle} seconds')
        time.sleep(idle)
